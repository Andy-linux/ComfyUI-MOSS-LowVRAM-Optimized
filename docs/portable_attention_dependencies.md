# 可选注意力依赖与安装阶段准备

默认使用主节点 **Prepare / check selected configuration** 查看环境与准备范围。
环境检查是只读操作，不下载二进制、不安装包、不运行GPU workload。维护者也可显式使用
`tools/check_environment.py`。默认 `auto` 使用PyTorch SDPA，不依赖Sage或Turing。

可选资产只在Manager的 `install.py` 阶段准备。Windows Manager可能到下次Restart才执行；
UI排队成功不等于依赖安装成功。安装不替换Torch/CUDA/Python，运行路由和推理不执行pip。
完整固定策略见 [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)。

## SM89 Flash

如存在私有 `runtime/vendor/xformers034cu128`，原显式xformers Flash FwOp与DLL句柄
管理路径保留；加载冲突明确报错。版本来自实际模块，不把目录名中的CUDA提示当作测量。
源码包不分发该私有binary，也不要求新用户为了auto安装它。

没有私有binary时，显式Flash使用PyTorch的 `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`，
禁止math/efficient/cuDNN替代；缺编译支持或不支持当前dtype/layout时报错，用户应保持auto
或使用兼容宿主。SM89、THD D128、相同FP16/BF16 Q/K/V、单完整dense段等限制仍然适用。
编译存在不代表任意shape、画质或性能已验证。

## SM89 SageAttention

精确Windows x64 / Python3.12 / Torch2.10.0+cu130 / CUDA13.0 / SM89宿主会在Manager
安装阶段尝试原发行固定post6 wheel，16,656,067 bytes、SHA256
`1635283f5c01ec3cda58a784d0d7eabbcaffaf9511d1b263db4750e1ed7958bb`。
目标为插件内 `runtime/vendor/sageattn220/`，复用固定helper的身份、ABI及staging校验，
不改全局site-packages。其他宿主跳过并给理由；下载/校验/原生导入失败只报warning，保持auto。

已有私有安装优先，否则可使用用户已安装的兼容Sage。模块必须提供所选FP16-PV/FP8-PV等
接口；显式Sage失败不会偷偷切换其他后端，auto也不会被暗中替换成Sage。
精确host的安装及fresh native import已通过，不代表每一种Sage布局/质量/速度通过。

## SM75 Turing Flash

用户已授权将自编Turing bundle通过独立GitHub Release分发。源码/Registry代码包不内嵌
其binary；真实本地ZIP与17成员清单已完成，固定Release目标和bytes/hash均已填入
`optional_binary_manifest.json`。Release 423709338已公开、非draft；匿名完整HTTP200下载
确认43,707,834 bytes及固定SHA256，与远端digest一致，清单已启用public。
清单固定发布URL、整体bytes/SHA256、逐成员hash、source commit与真实license/build/NOTICE记录。

源commit `9ef98fcb506bb1e2fe3cece50935e2935bf6b124` 未确认仓库LICENSE/许可声明；
CUTLASS/NVIDIA原文保留，不能据此声称整个Turing实现或组合binary已获某种开放许可。
用户分发授权不改变此事实。原始源码/构建记录经去除私有路径、GPU标识等信息后随独立资产保留。

安装与native probe仅覆盖Windows x64 / Python3.12 / Torch2.10.0+cu130 / CUDA13.0 / SM75，
SM89不安装/导入此包。adapter为推理用FP16、单dense完整段、THD H24/D128；不承诺BF16。
整体ZIP、逐文件、来源/许可记录检查在写入前完成，拒绝穿越、链接、未知文件和覆盖已有冲突目录。
安装后只在独立进程导入扩展和接口；没有执行GPU kernel。

文件匹配为installed；安装阶段原生导入通过可报告available；这些不是新增SM75图像/质量/速度
验证。历史scope与原始回执保留，不从SM89或auto验证推导Turing Flash通过。失败保持auto。
