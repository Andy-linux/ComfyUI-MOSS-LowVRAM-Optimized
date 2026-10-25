"""Pure request-boundary selection among qualified arithmetic implementations."""
from dataclasses import replace

def kernel_plan(config, latent_hw, positive_tokens, negative_tokens, sm, *, conditioning_eligible=False):
    image_tokens=int(latent_hw[0])*int(latent_hw[1])
    joint_tokens=image_tokens+max(int(positive_tokens),int(negative_tokens))
    changes={"qkv_backend":"shared" if config.qkv_backend=="auto" else config.qkv_backend,
             "ffn_tile":0 if config.ffn_tile==-1 else config.ffn_tile}
    if config.candidate in ("base_fp8","sft_fp8"):
        if sm<89:
            raise ValueError("Stored FP8 inference requires SM89+; current full workflow validation is on SM89")
        low_attention=config.attention.startswith("sage_") or config.attention=="ck_int8"
        group=(8 if low_attention and config.profile in ("4g","6g") else 0) if config.attention_head_group==-1 else config.attention_head_group
        changes.update(kernels="baseline",qkv_backend="shared",projection_tile=0,
                       qkv_reuse_input=False,rope_inplace=False,single_frame=False,
                       ffn_lifetime=False,ffn_decode_reuse=False,writer_epilogue=False,
                       attention_head_group=group)
        if config.ffn_tile==-1:
            changes["ffn_tile"]=2048 if config.profile=="4g" or joint_tokens>5894 else 0
        if config.profile=="4g" and config.resident_blocks<0:
            changes["resident_blocks"]=20
        effective=replace(config,**changes)
        return effective,{"image_tokens":image_tokens,"positive_tokens":int(positive_tokens),
            "negative_tokens":int(negative_tokens),"maximum_joint_tokens":joint_tokens,"SM":sm,
            "reason":"FP8 stored bases; original FP8 operators with bounded FFN and byte-payload placement",
            "qualification":"format-specific integration; R4 INT8 capacity timings do not transfer",
            "effective":changes}
    # Base uses the identical INT8 payload geometry, with its own checkpoint.
    # This reuse selects a capacity plan; it does not imply SFT image quality.
    if config.candidate=="base_w8_int8c":
        effective,receipt=kernel_plan(replace(config,candidate="w8_int8c"),latent_hw,
            positive_tokens,negative_tokens,sm,conditioning_eligible=conditioning_eligible)
        receipt["checkpoint_variant"]="base"
        receipt["reason"]+="; Base same-shape INT8 checkpoint"
        return replace(effective,candidate=config.candidate),receipt
    head_group=config.attention_head_group
    if head_group==-1:
        low_precision=config.attention.startswith("sage_") or config.attention=="ck_int8"
        head_group=8 if sm==89 and low_precision and (config.profile=="4g" or (config.profile=="6g" and config.candidate=="w8_int8c")) else 0
        changes["attention_head_group"]=head_group
    reason=("Conservative shared QKV" if config.qkv_backend=="auto" else "Explicit QKV backend="+config.qkv_backend)+"; original full-token FFN unless explicitly tiled"
    if config.qkv_backend=="auto" and sm==75:
        changes["kernels"]="v3_shared" if config.profile=="4g" else "native_shared"
        if config.profile=="4g":
            changes.update(single_frame=True,ffn_lifetime=True)
            reason="SM75 4g capacity: shared QKV and register-up FFNv3; native FFN exceeds the measured small-image budget"
        else:
            reason="SM75 original native FFN and shared QKV; fused register alternatives were slower in measured microbenchmarks"
    if config.qkv_backend=="auto" and config.profile=="4g" and config.candidate in ("q5_ridge_gptq","q6_direct") and sm in (89,75) and joint_tokens>5894:
        if joint_tokens>10206:
            raise ValueError("4 GB automatic capacity path is qualified up to 10206 joint tokens; choose a larger budget or smaller image")
        changes.update(qkv_backend="register3",qkv_reuse_input=True,rope_inplace=True,
                       single_frame=True,ffn_lifetime=True,projection_tile=2048)
        if config.ffn_tile==-1:changes["ffn_tile"]=2048
        weight="W5" if config.candidate=="q5_ridge_gptq" else "W6"
        reason="SM"+str(sm)+" "+weight+" large joint sequence: qualified private-input Q reuse, original in-place RoPE, single-frame plan and output projection tiles"
    if config.profile=="4g" and config.candidate=="w8_int8c" and sm==89:
        changes.update(single_frame=True,ffn_lifetime=True,projection_tile=2048)
        if joint_tokens>5894 and config.ffn_tile==-1:changes["ffn_tile"]=2048
        reason="SM89 W8 original INT8 payload streaming; fused FFN and M-tiled native output projection"
    if config.profile=="6g" and config.candidate=="w8_int8c" and sm==89 and joint_tokens>5894:
        changes.update(single_frame=True,ffn_lifetime=True)
        if config.ffn_tile==-1:changes["ffn_tile"]=2048
        reason="SM89 W8 6g long sequence: original full INT8 payload with bounded FFN activations"
    if config.profile=="4g" and config.candidate in ("q6_direct","w8_int8c") and sm==89 and config.resident_blocks<0:
        prefix=33 if config.candidate=="q6_direct" else 23
        # RequestAdapter retains BOTH text branches. A maximum joint-token
        # bound alone does not bound the simultaneous adapter storage:
        # 2 bytes * (4096 + 3072) * (positive_tokens + negative_tokens).
        # Preserve the measured projection/FFN stack when spending its margin.
        short_stack=(conditioning_eligible and joint_tokens<=4362
                     and int(positive_tokens)+int(negative_tokens)<=285
                     and config.te_memory in ("ring","auto")
                     and str(config.te_weights).replace("\\","/").rsplit("/",1)[-1]=="moss-image-qwen35-int8-convrot.safetensors"
                     and config.kernels=="v3_shared" and changes["ffn_tile"]==0
                     and (config.candidate=="w8_int8c" or changes["qkv_backend"]=="shared")
                     and not (config.writer_epilogue or config.ffn_decode_reuse
                              or config.qkv_reuse_input or config.rope_inplace))
        if config.attention=="flash" and short_stack:
            prefix+=2
        # W8 p26: measured end-to-end against p23 with identical conditioning
        # and final latent. W6 p35 saved only ~0.13 s; keep its p33 default.
        # Do not extrapolate this FP16/H8 qualification to other Sage modes.
        if (config.candidate=="w8_int8c" and config.precision=="fp16_fixed"
                and config.attention=="sage_fp8_rot" and head_group==8 and short_stack):
            prefix=26
        if (config.attention.startswith("sage_") or config.attention=="ck_int8") and not (conditioning_eligible and head_group==8):prefix-=2
        if config.candidate=="q6_direct" and joint_tokens>5894 and not conditioning_eligible:prefix-=2
        changes['resident_blocks']=prefix
        reason+="; qualified DiT payload prefix="+str(prefix)
    effective=replace(config,**changes)
    receipt={"requested_qkv_backend":config.qkv_backend,"requested_ffn_tile":"auto" if config.ffn_tile==-1 else config.ffn_tile,
             "image_tokens":image_tokens,"positive_tokens":int(positive_tokens),"negative_tokens":int(negative_tokens),
             "maximum_joint_tokens":joint_tokens,"SM":sm,"reason":reason,
             "conditioning_table_eligible":conditioning_eligible,
             "effective":{k:getattr(effective,k) for k in ("kernels","qkv_backend","ffn_tile","qkv_reuse_input","rope_inplace","single_frame","ffn_lifetime","projection_tile","sdpa_capacity_fallback","resident_blocks","attention_head_group")}}
    return effective,receipt
