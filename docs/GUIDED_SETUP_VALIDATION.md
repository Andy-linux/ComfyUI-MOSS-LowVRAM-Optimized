# 统一00工作流验证：候选环境、模型复用与安装边界

本页记录新增00工作流的实际验证，和历史01–07回执分别保存。
下列结果不等于新机器从公开Registry完成缺节点安装，亦不表示模型首次网络下载已经测试。
GitHub源码与Turing Release已公开，Registry尚未发布。两份Base/SFT FP8 row已公开并固定于[R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8/tree/d8a0b2f46a79770fe711bf5322cf6cb3219c8d41)，合计10,349,759,704 bytes。已核对两文件精确大小、远端LFS SHA256与本地完整文件SHA256，并读回小型配套文件；没有重新下载约10GB远端权重做全流SHA256核验。


匿名访问记录 `hf-public-anonymous-01.json`（Windows bench）进一步确认固定R1可公开读取：
客户端未发送Authorization；metadata GET返回200，`private=false`、`sha`精确等于上述R1，
共7个文件。Base与SFT的HEAD均返回200，Content-Length分别为5,174,879,864与5,174,879,840 bytes。
本轮未使用Range fallback，也未进行远端权重全流hash；这些HTTP结果不等于Manager已下载模型。

## 候选服务上的GUI准备与正常步数生成

Windows候选服务上的唯一用户工作流 `example_workflows/00_moss_lowvram.json`
已通过 **19项GUI检查**。准备作业
`c0f6b40e6dc84e04bc19dc374ac81572` 全部ready，复用三份预存模型：

| 资产 | 实际复用场景 | 模型网络下载 |
| --- | --- | --- |
| SFT W5 DiT | 改名文件位于注册模型子目录 | 0 |
| Sumomo INT8 TE v2 | 子目录中的别名文件 | 0 |
| 原MOSS VAE | 注册的额外模型根目录 | 0 |

该结果验证GUI准备可识别、核验并复用这些已有模型位置，没有为了推荐文件名重复下载。
它没有测试缺文件情况下的实际大模型网络传输，也没有把用户的本地路径发布为模型来源。

准备完成后，实际prompt `94fd4b29-a191-4ec0-977b-2f01326b2d6d` 成功：

| 项目 | 实际值 |
| --- | --- |
| 系列/DiT | 默认SFT W5 |
| TE计算 | W8A16 |
| 分辨率 | 1024 × 1024 |
| 采样步数 | 16 |
| Seed | 314159265 |
| 输出 | `MOSS/00_moss_lowvram_00001_.png` |
| 日志执行区间 | 25.76秒 |

25.76秒只记录这次日志执行区间，不解释为新机器首次安装/准备总时间，也不作为所有显卡、
模型或后端的速度承诺。一次生成成功不自动构成跨模式画质认证。

## 默认宿主下的FP32 VAE修复：阴性与阳性GPU探针

修复前的 `canonical0f7` 候选在宿主默认BF16条件下运行真实探针，
`Session.decode` 明确失败：`MOSS VAE requires its FP32 latent input boundary`。
这是曾依赖全局 `--fp32-vae` 才能满足VAE边界的实际缺陷，不是安装或模型下载失败。

修复提交 `bab2e422fd240cbfc107bb75da15fb7d9b8c7ce8` 已推送。
同一真实GPU探针在未传 `--fp32-vae`、也未传 `--disable-dynamic-vram` 的条件下通过：
256 × 256的zero-latent解码输出全部有限，VAE的 **62,373,347个浮点tensor元素全部为FP32**。
首次创建与缓存工厂路径均观察到FP32，无参数调用和 `disable_dynamic=True` 调用也均保持FP32。
修复范围为 `session.py` 与小型FP32缓存工厂，未修改BF16函数。

本轮三处实际 `patcher.is_dynamic()` 观察均为 `false`，因此只证明所测非dynamic路径的
默认宿主FP32 VAE边界修复；没有验证dynamic patcher或多GPU。
该256²边界探针与上方1024²/16步候选生成分别记录，不能据此把后者追溯改写为
移除所有测试启动参数后的正常步数生成验证。

## cold03：最小默认启动与独立空缓存基线

独立Manager服务的cold03启动仅使用 `main.py` 的listen、port与enable-manager参数，
已移除先前测试用的FP32/CUDA/preview启动选项及两项allocator环境覆盖。
HTTP确认ComfyUI0.41、SM89，宿主默认自动选择 `cudaMallocAsync`。

该子进程为HF、Triton、Torch、Inductor与CUDA分别使用独立缓存，启动前各类缓存文件数均为0；
启动后HF/Triton/Torch/Inductor仍为0，CUDA的2个文件由本次启动产生。
仅在该子进程禁用隐式HF鉴权、移除HF token环境变量，以及 `CUDA_PATH`、`CUDA_PATH_V12_6`
和两项CUDA/MSVC相关PATH条目；原用户缓存、全局环境与正在运行的上传任务均未改动。
这描述测试隔离条件，不是用户运行要求。

`/object_info` 无MOSS节点，Triton与accelerate仍缺失，模型目录中大于1MB的文件为0，queue为空。
对应原始记录为 `manager-server-process-cold-03.json` 与 `manager-cold-baseline-03.json`。
这轮只建立更严格的空缓存、默认启动基线；**尚未完成Manager安装本包、模型下载或出图**，
不能作为这些步骤的PASS；HF权重公开验证是独立结果，Registry发布与实际Manager闭环仍待完成。

## 独立证据层次

| 验证层 | 已观察结果 | 没有由此证明的内容 |
| --- | --- | --- |
| 干净Python依赖准备 | Windows Python3.12.13 / Torch2.10+cu130 / Manager4.2.2，起初无Triton/accelerate，requirements与install.py退出0；固定Sage隔离准备及fresh native import通过，Torch/Triton版本未替换 | 不是从Manager UI发现缺失节点后安装的记录，不是新装OS认证 |
| 新00界面 | 独立Chrome context run03先通过16项联动/九文件Original TE/metadata/24值序列化检查；候选服务后续19项GUI检查及三模型复用ready | 前一run03没有download/prompt/Manager安装；后续复用场景也没有模型网络下载 |
| 候选服务端到端 | 上述1024²、16步真实prompt成功并输出PNG | 不等于空Manager基线的节点发现、源码下载、依赖安装和重启闭环 |
| 共享FP8 | Base/SFT × FP8/16-bit × 两次，共8请求，512²/2步/解码；263层q/scale指针、hash、bytes不变，8项CPU测试0skip；4070TiSUPER、cudaMallocAsync | 不是全步数画质或普遍速度benchmark，也不覆盖SM75 |
| 单元与冻结helper | 此前delta04 Windows137/137通过、0skip，GUI相关32项通过；冻结Turing helper独立8项focused通过，真实17成员ZIP纯校验通过 | focused不是再次跑完137项；ZIP检查不运行SM75 GPU工作负载 |
| 真正空基线Manager安装 | cold03最小默认启动、独立空缓存基线已建立；公开包安装、模型下载与出图仍待完成 | 不把预先放置候选插件、已有模型或直接执行install.py写成Manager新装成功 |

## 可选Turing资产与公共状态

[Turing独立Release](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/releases/tag/turing-sm75-cp312-cu130-v1) 已公开、非draft，Release ID423709338；root已核对远端asset digest。
Turing ZIP为43,707,834 bytes，SHA256：
`779e2bf85ae347a50200b372eb79481bce069a0392856cd09e9f0b8dbd0fcae7`。
Root随后不带Authorization匿名完整下载得到HTTP200，精确大小/hash均匹配；
独立清单已改public，严格匹配SM75宿主可在Manager安装阶段准备。公开资产不等于Manager闭环已完成。

资产保留真实source/build/NOTICE及原CUTLASS/NVIDIA许可文本。源Turing固定HEAD未确认仓库
LICENSE的事实继续披露，用户授权分发不被描述为补造上游许可。
冻结helper `ff162c…` 已在Windows另跑8项focused测试通过（该轮日志`turing-focused-07.log`），
并通过真实17成员ZIP的纯 `validate_archive` 检查。这不表示137项全套又重新运行；
这些资产/hash/单元测试也不是新增SM75实卡采样或Flash图像质量验证。

## 发布前仍需完成

公开资产与Registry版本实际可访问后，在没有本插件、Triton、accelerate和模型的独立
Manager基线中，真实导入带有效cnr_id/ver的00，验证Missing Nodes → Manager安装 →
Restart → 节点注册/依赖状态 → GUI准备/生成。保留真实queue与重启后结果，分别说明
模型复用、真正网络下载以及可能的Registry提取/索引等待，不以当前候选服务PASS代替。

旧 `docs/validation/` 与历史发布候选回执保留原数值、scope和SHA，不重写为本页的新结果。
