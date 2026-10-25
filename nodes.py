"""Normal ComfyUI nodes for frozen packed SFT inference and explicit numerical modes."""
import logging
import secrets
import time
import comfy.samplers
from .runtime.session import MossSession,RuntimeConfig
from .runtime.resources import allocator_budget

_MODELS={"SFT W5":"q5_ridge_gptq","SFT W6":"q6_direct","SFT W8":"w8_int8c",
         "Base W8A8 INT8 ConvRot":"base_w8_int8c",
         "Base Kohaku FP8":"base_fp8","SFT Kohaku FP8":"sft_fp8"}
_PROFILES={"4 GB":"4g","6 GB":"6g","8 GB":"8g","12 GB":"12g","16 GB":"16g","unlimited":"unlimited"}
_ATTENTION={"auto":"auto","Flash FP16":"flash","Flash FP16/BF16":"flash",
            "cuDNN dense":"cudnn","Sage QK8 PV16":"sage_fp16",
            "Sage QK8 PV16 + H128":"sage_fp16_rot","Sage QK8 / PV FP8":"sage_fp8",
            "Sage QK8 / PV FP8 fast":"sage_fp8_fast","Sage QK8 / PV FP8 fast + H128":"sage_fp8_rot",
            # Early R4 PNG/workflow values remain loadable after clearer labels.
            "Sage FP8":"sage_fp8","Sage FP8 fast":"sage_fp8_fast","Sage FP8 fast + H128":"sage_fp8_rot",
            "CK INT8":"ck_int8"}
_TEXT={
    # Historical widget values retain their historical A16 numerical path.
    "INT8 ConvRot":("te_int8","auto","fp16"),
    "W4A8 GPTQ":("te_w4","auto","fp16"),
    "INT8 ConvRot · W8A16":("te_int8","w8a16","fp16"),
    "INT8 ConvRot · W8A8":("te_int8","w8a8","fp16"),
    "W4 GPTQ · W4A8":("te_w4","w4a8","fp16"),
    "Original BF16":("te_original","original","bf16"),
    "Original FP16":("te_original","original","fp16"),
}

class MOSSOptimizedSessionLoader:
    CATEGORY="Kohaku MOSS/Optimized"
    RETURN_TYPES=("MOSS_SESSION",)
    FUNCTION="load_session"
    DESCRIPTION="MOSS Base/SFT：保留原 W5/W6/W8，支持 Base ConvRot W8A8 与 Kohaku FP8。TE 的存储格式和实际计算精度分开选择；当前不支持 LoRA。"

    @classmethod
    def INPUT_TYPES(cls):
        from .runtime.assets import checkpoint_choices
        return {"required":{
            "model":(list(_MODELS),{"tooltip":"W5/W6/W8 使用原整数量化码；位数越高，权重占用越大，画面可能变化。4 GB 的 W6/W8 逐层搬运部分权重，耗时受 PCIe 影响。"}), 
            "profile":(list(_PROFILES),{"default":"6 GB","tooltip":"4/6/8/12/16 GB 对应 3.5/5.5/7.5/11.5/15.5 GiB 进程分配限额。cudaMallocAsync 限制申请字节，内存池 reserved 可更大；不是整卡专用显存硬上限。不改变尺寸和步数。"}), 
            "text_encoder":(list(_TEXT),{"tooltip":"旧 INT8 ConvRot 实际为 W8A16；旧 W4A8 GPTQ 保留原 W4存储+A16 路径。新带 W8A8/W4A8 标签的选项执行真实 INT8 GEMM，数值会变化。Original 加载原始浮点权重，auto 内存策略使用 ring。所有 conditioning 输出为 FP16。"}), 
            "attention":(list(_ATTENTION),{"tooltip":"auto/Flash/cuDNN 不额外量化 Q/K/V。Sage 的 QK8 为 Q/K INT8；PV16 路径的 V 使用 FP16，PV FP8 量化 P/V。fast 使用专用 V 缩放和混合累加；H128 匹配旋转可能减轻部分误差。Sage/CK 可能改变文字、人脸和构图，不能视为普遍无损。本轮新增 Sage/CK/cuDNN 适配限 SM89。"}), 
            "qkv_backend":(["auto","shared","native","register","register3"],{"tooltip":"auto 按显卡和工作集选路径。shared 共享 A8 量化；native 合并投影；register 融合写回；register3 使用独立 Q/K/V 输出。W5/6G/Flash 实测 register3 小幅提速，其他组合未普遍验证；4G 长图请保持 auto，以启用完整容量策略。"}), 
            "ffn_tile":(["auto",0,2048],{"tooltip":"auto 优先整段计算，容量紧张才分块；0 强制整段；2048 降低激活峰值，通常增加耗时。"}), 
            "host_arena":("BOOLEAN",{"default":True,"tooltip":"复用 W5/W6 packed 存储，或 W8/6G、8G 的原 INT8 CPU arena。W8 arena 约 4.43 GiB 系统内存，首次组装有成本；必要 GPU 上传仍保留。4G 流式路径总是复用原码。"}),
        },"optional":{
            "adapter_cache":("BOOLEAN",{"default":False,"tooltip":"精确复用相同编码与尺寸的 adapter 输出；CPU 最多 64 MiB。重复抽图减少权重搬运，新提示词仍完整编码。"}),
            "conditioning_memory":("BOOLEAN",{"default":False,"tooltip":"4 GB、Euler、1–64 步时预计算原调制结果，采样少占约 162 MiB；增加准备时间。容量可选项，其他配置自动走原路径。"}),
            "precision":(["fp16_fixed","bf16"],{"default":"fp16_fixed","tooltip":"实际 DiT 数值路径：fp16_fixed 保留原修正；bf16 改变非量化参数及计算、存回边界，不应用 FP16 修正。W5/W6/W8 整数权重码不变。两者显存相近、画面可能变化；BF16 需 SM80+。"}),
            "attention_head_group":(["auto",0,8,12,24],{"default":"auto","tooltip":"auto 为 4G、W8/6G 的 Sage/CK 选 8 头，其余不分组。0/24 不分组；8/12 保留完整 K/V，省临时显存但增加调度。Flash/auto 注意力仅支持不分组。"}),
            "vae_precision":(["fp32","bf16"],{"default":"fp32","tooltip":"FP32 为原解码；BF16 减少解码显存并提速，会有小像素差异。两者保留原 FP32 latent 反归一化；BF16 需 SM80+。"}),
            "te_memory":(["auto","ring","partial","resident"],{"default":"auto","tooltip":"auto 按已验证预算选择。ring 逐层搬运；partial 保留部分层；resident 保留约 6.46 GiB INT8 TE 主体，embedding 仍在 CPU。过多驻留可能使解码卸载模型，反而变慢。"}),
            "vae_admission":("BOOLEAN",{"default":False,"tooltip":"使用同运行库、已测尺寸的 VAE 工作区估计，减少不必要的模型卸载。未知尺寸保留原估计；预算不增加，容量不足仍官方分块。"}),
            "resident_blocks":("INT",{"default":-1,"min":-1,"max":40,"tooltip":"高级：-1 自动；0–40 强制 DiT 常驻前缀并用双缓冲搬运其余层。更多常驻减少 PCIe 流量，也挤占激活空间；一般保持 -1。"}),
            "payload_pinning":(["staging","suffix"],{"default":"staging","tooltip":"staging 省锁页：流式用小双缓冲，W8 完整 arena 用普通内存整段上传。suffix 另缓存最多 2.5 GiB 活动锁页副本，4G 重复搬运有利；首次准备更重。卸载释放本会话缓存引用。"}),
            "te_resident_gib":("FLOAT",{"default":3.0,"min":0.25,"max":6.0,"step":0.25,"tooltip":"仅 te_memory=partial 生效：TE 常驻前缀的权重上限，单位 GiB。另需后缀双缓冲与激活空间；不增加整个会话显存预算。仅 SM89 已适配。"}),
            "dit_checkpoint":(checkpoint_choices("diffusion_models"),{"default":"auto","tooltip":"auto 按模型类型查找标准文件名或本机映射。显式选择时须与 Base/SFT 和量化格式匹配；不会自动下载。"}),
            "te_checkpoint":(checkpoint_choices("text_encoders"),{"default":"auto","tooltip":"auto 按 TE 模式查找；原始 HF 分片目录可放 models/text_encoders/MOSS-Image-Base/text_encoder，配套 tokenizer 放其同级。"}),
            "vae_checkpoint":(checkpoint_choices("vae"),{"default":"auto","tooltip":"MOSS/Flux2 latent 兼容 VAE；默认使用下载清单中的原 MOSS VAE。"}),
        }}

    def __init__(self):
        self.session=None

    def load_session(self,model,profile,text_encoder,attention,qkv_backend,ffn_tile,host_arena,adapter_cache=False,conditioning_memory=False,
                     precision="fp16_fixed",attention_head_group=0,vae_precision="fp32",te_memory="auto",vae_admission=False,resident_blocks=-1,payload_pinning="staging",te_resident_gib=3.0,
                     dit_checkpoint="auto",te_checkpoint="auto",vae_checkpoint="auto"):
        if attention not in _ATTENTION:
            raise ValueError("未知注意力后端："+str(attention))
        backend=_ATTENTION[attention]
        head_group=-1 if attention_head_group=="auto" else int(attention_head_group)
        if backend in ("auto","flash") and head_group not in (-1,0,24):
            raise ValueError("Flash/auto 不支持 head-group；请选择 0，或切换到支持分组的后端。")
        te_asset,te_mode,te_dtype=_TEXT[text_encoder]
        config=RuntimeConfig(candidate=_MODELS[model],profile=_PROFILES[profile],
             te_weights=te_checkpoint,te_asset=te_asset,te_mode=te_mode,te_compute_dtype=te_dtype,
             dit_weights=dit_checkpoint,vae_name=vae_checkpoint,attention=backend,precision=precision,attention_head_group=head_group,
             qkv_backend=qkv_backend,ffn_tile=-1 if ffn_tile=="auto" else int(ffn_tile),host_arena=host_arena,
             te_memory=te_memory,vae_precision=vae_precision,vae_admission=bool(vae_admission),resident_blocks=int(resident_blocks),
             payload_pinning=payload_pinning,te_resident_gib=float(te_resident_gib),
             resident_vae_stage_budget=profile=="unlimited" and te_memory=="auto" and vae_precision=="fp32" and not vae_admission,
             adapter_cache=bool(adapter_cache),conditioning_schedule=bool(conditioning_memory))
        if self.session is None or self.session.config!=config:
            if self.session is not None:
                self.session.close()
            self.session=MossSession(config)
            logging.info("[KohakuMOSS lifecycle] new session profile=%s candidate=%s device=%s",config.profile,config.candidate,config.device)
        return (self.session,)

class MOSSOptimizedPairedTextEncode:
    CATEGORY="Kohaku MOSS/Optimized"
    RETURN_TYPES=("CONDITIONING","CONDITIONING")
    RETURN_NAMES=("positive","negative")
    FUNCTION="encode"
    DESCRIPTION="正负提示词逐层配对处理，共享权重搬运，保留各自完整序列。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "session":("MOSS_SESSION",{"tooltip":"连接模型与显存方案节点。"}),
            "positive":("STRING",{"multiline":True,"dynamicPrompts":False,"tooltip":"希望生成的内容；完整编码，不自动截断。更长文本增加编码与采样开销。"}),
            "negative":("STRING",{"multiline":True,"dynamicPrompts":False,"default":"","tooltip":"希望避免的内容；保留完整负分支，参与 CFG 与 SLG。"}),
        }}

    def encode(self,session,positive,negative):
        with allocator_budget(session,operation="TE"):
            return session.encode_pair(positive,negative)


def _seed_metadata(seed, unique_id, prompt, extra_pnginfo, was_random):
    if unique_id is None:
        return
    key=str(unique_id)
    if prompt is not None and key in prompt:
        prompt[key]["inputs"]["seed"]=seed
    if extra_pnginfo is not None:
        extra_pnginfo.setdefault("moss_actual_seeds",{})[key]=seed
        workflow=extra_pnginfo.get("workflow",{})
        for node in workflow.get("nodes",[]):
            if str(node.get("id"))==key and node.get("type")=="MOSSOptimizedSampler":
                values=node.get("widgets_values",[])
                if values:
                    values[0]=seed
                    if was_random and len(values)>1 and values[1] in ("fixed","increment","decrement","randomize"):
                        values[1]="fixed"
                named=node.get("widgets_values_named")
                if isinstance(named,dict) and "seed" in named:
                    named["seed"]=seed
                    if was_random and "control_after_generate" in named:
                        named["control_after_generate"]="fixed"


class MOSSOptimizedSampler:
    CATEGORY="Kohaku MOSS/Optimized"
    RETURN_TYPES=("LATENT",)
    FUNCTION="sample"
    DESCRIPTION="使用会话选择的实际数值路径；基线 16 步、CFG 3、Euler / sgm_uniform、shift 5、SLG 22。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "session":("MOSS_SESSION",{"tooltip":"复用模型会话及其显存预算。"}),
            "positive":("CONDITIONING",{"tooltip":"连接配对编码节点的 positive。"}),
            "negative":("CONDITIONING",{"tooltip":"连接配对编码节点的 negative；默认保留完整负分支。"}),
            "latent_image":("LATENT",{"tooltip":"连接 EmptyFlux2LatentImage，控制尺寸；当前 batch 为 1。"}),
            "seed":("INT",{"default":-1,"min":-1,"max":0xffffffffffffffff,"control_after_generate":True,"tooltip":"-1 每次生成新随机种子；非负数固定。实际种子写入 PNG，可拖回工作区复现。"}),
            "steps":("INT",{"default":16,"min":1,"max":10000,"tooltip":"去噪步数；16 为验证基线。增加步数增加耗时，减少步数会改变质量。"}),
            "cfg":("FLOAT",{"default":3.0,"min":0.0,"max":100.0,"step":0.1,"tooltip":"提示词引导强度；基线 3。改变它会改变画面，并可能改变分支计算量。"}),
            "sampler_name":(comfy.samplers.KSampler.SAMPLERS,{"default":"euler","tooltip":"采样算法；基线 Euler。其他算法可能增加模型调用次数或改变结果。"}),
            "scheduler":(comfy.samplers.KSampler.SCHEDULERS,{"default":"sgm_uniform","tooltip":"时间步分布；基线 sgm_uniform。改变分布会改变生成轨迹。"}),
            "shift":("FLOAT",{"default":5.0,"min":0.0,"max":100.0,"step":0.1,"tooltip":"Flow 时间调度偏移，基线 5；不同值会改变细节、布局及收敛。"}),
            "slg_layer":([22],{"tooltip":"负分支跳过第 22 层；保持当前已验证的 MOSS SLG 语义。"}),
        },"hidden":{"unique_id":"UNIQUE_ID","prompt":"PROMPT","extra_pnginfo":"EXTRA_PNGINFO"}}

    @classmethod
    def IS_CHANGED(cls,seed,**kwargs):
        return float("nan") if seed==-1 else seed

    def sample(self,session,positive,negative,latent_image,seed,steps,cfg,sampler_name,scheduler,shift,slg_layer,
               unique_id=None,prompt=None,extra_pnginfo=None):
        import latent_preview
        was_random=seed==-1
        seed=secrets.randbits(53) if was_random else int(seed)
        if not 0<=seed<=0xffffffffffffffff:
            raise ValueError("种子必须为 -1 或非负 64 位整数。")
        _seed_metadata(seed,unique_id,prompt,extra_pnginfo,was_random)
        with allocator_budget(session,operation="DiT"):
            result=session.sample(positive,negative,latent_image,seed,steps=steps,cfg=cfg,
                     sampler_name=sampler_name,scheduler=scheduler,shift=shift,slg_layer=slg_layer,
                     callback_factory=latent_preview.prepare_callback)
        session.last["actual_seed"]=seed
        logging.info("[KohakuMOSS seed] actual_seed=%d",seed)
        return {"ui":{"actual_seed":[str(seed)]},"result":(result,)}

class MOSSOptimizedVAEDecode:
    CATEGORY="Kohaku MOSS/Optimized"
    RETURN_TYPES=("IMAGE",)
    FUNCTION="decode"
    DESCRIPTION="使用会话指定的 FP32/BF16 VAE 解码。显存不足时官方分块回退；精度或分块变化可能改变局部像素。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"session":("MOSS_SESSION",{"tooltip":"沿用会话的 VAE 与显存预算。"}),
                            "samples":("LATENT",{"tooltip":"连接采样输出；完整解码优先，容量不足时自动分块。"})}}

    def decode(self,session,samples):
        with allocator_budget(session,operation="VAE"):
            return (session.decode(samples),)

class MOSSOptimizedReleaseSession:
    CATEGORY="Kohaku MOSS/Optimized"
    RETURN_TYPES=()
    OUTPUT_NODE=True
    FUNCTION="release"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"session":("MOSS_SESSION",{"tooltip":"只释放此 MOSS 会话。"}),
                            "enabled":("BOOLEAN",{"default":False,"tooltip":"开启后在图像完成时释放会话。日常连续出图保持关闭，可改用维护节点按钮。"})},
                "optional":{"images":("IMAGE",{"tooltip":"连接图像可让清理等待解码完成。"})}}

    @classmethod
    def IS_CHANGED(cls,enabled=False,**kwargs):
        return float("nan") if enabled else False

    def release(self,session,enabled=False,images=None):
        if enabled:
            session.close()
        return {}


class MOSSOptimizedControls:
    CATEGORY="Kohaku MOSS/Optimized"
    RETURN_TYPES=()
    OUTPUT_NODE=True
    FUNCTION="run"
    DESCRIPTION="按钮通过普通 ComfyUI 队列执行。预热使用当前提示词和尺寸运行一小步及 VAE，不保存图片。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "session":("MOSS_SESSION",{"tooltip":"要预热或卸载的 MOSS 会话。"}),
            "action":(["idle","prewarm","unload"],{"default":"idle","tooltip":"日常保持 idle（待机），使用下方按钮提交预热或卸载。卸载释放本会话及进程空闲锁页缓存，活跃缓冲保留；下次需重新加载。按钮不改变此默认值。"}),
        },"optional":{
            "positive":("CONDITIONING",{"tooltip":"预热用的完整正向编码；连接配对编码节点。"}),
            "negative":("CONDITIONING",{"tooltip":"预热用的完整负向编码；卸载按钮不运行编码。"}),
            "latent_image":("LATENT",{"tooltip":"预热当前图像尺寸；更换尺寸或后端后可以再次预热。"}),
        }}

    @classmethod
    def IS_CHANGED(cls,action="idle",**kwargs):
        return float("nan") if action!="idle" else "idle"

    def run(self,session,action="idle",positive=None,negative=None,latent_image=None):
        if action=="idle":
            return {}
        if action=="unload":
            session.close()
            cleanup=session.last.get("host_cache_release",{})
            status="已卸载本 MOSS 会话与 CPU 权重缓存；下次生成按需加载。"
            if cleanup.get("status")=="released_idle":status+=" 已释放进程空闲锁页缓存。"
            return {"ui":{"moss_status":[status],"moss_cleanup":[cleanup]}}
        if action!="prewarm":
            raise ValueError("未知维护操作。")
        if positive is None or negative is None or latent_image is None:
            raise ValueError("预热需要连接正向、负向编码和 latent_image 尺寸。")
        started=time.perf_counter()
        with allocator_budget(session,operation="prewarm"):
            latent=session.sample(positive,negative,latent_image,0,steps=1,cfg=3,
                                  sampler_name="euler",scheduler="sgm_uniform",shift=5,slg_layer=22)
            image=session.decode(latent)
            del image,latent
        elapsed=time.perf_counter()-started
        return {"ui":{"moss_status":[f"预热完成（采样与解码 {elapsed:.1f} 秒）。当前形状常用路径已预热；新参数仍可能触发编译。"]}}

NODE_CLASS_MAPPINGS={
    "MOSSOptimizedSessionLoader":MOSSOptimizedSessionLoader,
    "MOSSOptimizedPairedTextEncode":MOSSOptimizedPairedTextEncode,
    "MOSSOptimizedSampler":MOSSOptimizedSampler,
    "MOSSOptimizedVAEDecode":MOSSOptimizedVAEDecode,
    "MOSSOptimizedReleaseSession":MOSSOptimizedReleaseSession,
    "MOSSOptimizedControls":MOSSOptimizedControls,
}
NODE_DISPLAY_NAME_MAPPINGS={
    "MOSSOptimizedSessionLoader":"Kohaku MOSS · 模型与预算",
    "MOSSOptimizedPairedTextEncode":"MOSS · 正负提示词编码",
    "MOSSOptimizedSampler":"MOSS · 优化采样",
    "MOSSOptimizedVAEDecode":"MOSS · VAE 解码",
    "MOSSOptimizedReleaseSession":"MOSS · 出图后释放",
    "MOSSOptimizedControls":"MOSS · 预热与卸载",
}

