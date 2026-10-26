# MOSS LowVRAM Optimized 发布顺序与当前边界

## 当前身份与实际状态

- 正式包ID：`kohaku-moss-lowvram-optimized`；展示为 **MOSS LowVRAM Optimized**，使用KohakuMoss推理组件。
- GitHub目标：[Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized)。已公开，已确认源码提交 `523e1e68190d1a17138b2189ad24349e175d938d`；后续清单启用修改保留各自同步提交。
- Registry Publisher `xianbei`公开状态Active；Registry包尚未发布。此前本地加密凭证可读取/解密，不等于PAT服务端认证已通过；不借无必要写入试探认证。
- 两份Base/SFT FP8 row已公开并固定于[R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`](https://huggingface.co/Andylinux/MOSS-Image-Kohaku-FP8/tree/d8a0b2f46a79770fe711bf5322cf6cb3219c8d41)，合计10,349,759,704 bytes。已核对两文件精确大小、远端LFS SHA256与本地完整文件SHA256，并读回小型配套文件；没有重新下载约10GB远端权重做全流SHA256核验。一个row文件共享FP8/16-bit计算；旧独立weight-only仅legacy local，不是第三份上传。
- 用户已明确授权自编SM75 Turing bundle作为独立GitHub Release。Release 423709338已公开、非draft；root匿名HTTP200完整下载43,707,834 bytes与SHA779e2bf8…匹配远端digest，清单已启用public。源码包不内嵌binary。

根KohakuMoss LICENSE、NOTICE与原vendor/组件许可证字节保持不变；命名与组件scope继续保留。
上游Turing固定HEAD未确认仓库许可的事实不变，用户分发授权不被描述为上游开放许可。

## 1. 固定资产与安装策略

普通依赖在requirements与pyproject同步，包括 `diffusers>=0.41.0,<0.42`、
`transformers>=5.20.1,<6`、accelerate等。Manager安装阶段处理requirements和install.py；
环境检查与运行路由不会pip，不替换宿主Torch/CUDA/Python。

Windows精确自动准备组合为x64 Python3.12/Torch2.10+cu130，SM89对应固定Triton3.6与
公开Sage；SM75使用明确标记历史配对的Triton3.2，Turing仅在其独立真实清单完整后准备。
SM89不会下载/导入SM75 Turing。未知宿主给诊断，不盲装最新版，不为安装演示降低Manager安全级别。

Turing已公开的独立资产tag为 `turing-sm75-cp312-cu130-v1`，文件名
`moss-turing-flash-cp312-torch210-cu130-sm75-win_amd64.zip`。这些是已确认公开的固定资产名称；
匿名完整下载与远端digest核验均已通过。ZIP根直接对应turing_bundle，保留真实源码归档、构建记录、
portable MANIFEST、CUTLASS/NVIDIA原文、NOTICE及完整hash清单，去除私有路径/设备标识。
固定Turing源HEAD `9ef98fcb506bb1e2fe3cece50935e2935bf6b124` 无已确认仓库LICENSE；
不能把CUTLASS/NVIDIA许可扩大为整个组合binary的许可。

## 2. 保持已有模型来源，上传两份自有row

已有Base/SFT INT8、TE、VAE继续索引清单固定上游，不重传。
W5/W6继续使用 `Andylinux/MOSS-Image-SFT-W5A8-W6A8` 的revision
`bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`，保留其README、LOWBIT_MANIFEST和许可。

自有FP8仅发布Base/SFT两份row，权重提交R0为 `082aebf29d592d9a9f0ba61b64f780ad160ec72b`，
单次文档提交及readback得到R1 `d8a0b2f46a79770fe711bf5322cf6cb3219c8d41`，核验范围见[模型文件](../MODEL_FILES.md)。
GUI下载使用真实固定revision/bytes/hash；该HF结果不表示Registry或Manager安装闭环已完成。
不向已有W5/W6仓库随意追加，也不把本地旧weight-only作为第三份公开下载。

## 3. 先获得真实Registry身份，再导出唯一工作流

源码与Registry发布成功并确认实际可安装后，从该安装导出唯一
`example_workflows/00_moss_lowvram.json`，携带真实 `properties.cnr_id` 与 `ver`。
当前cloud00已有待发布cnr_id/ver metadata，不表示Registry已发布；发布安装后仍需核对再导出。保留六个旧class IDs并新增统一配置入口。
旧01–07只在 `tests/historical_workflows/` 保存为历史fixture，由.comfyignore排除Registry；
不是当前模板、下载入口或新用户分享方案。

只读确认使用节点详情、精确版本、版本列表与comfy-node映射。`GET /nodes/{id}/install`
会记录安装行为，不作为监控；`GET /security-scan`实际执行扫描，也不作为只读检查。
公开Active/downloadUrl、映射提取、搜索索引、Manager缓存、实际安装是分别验证的状态。
Pending/Flagged或索引延迟如实记录，不虚构等待时间，也不自动重发相同版本。

## 4. 实际新用户GUI闭环

在干净隔离ComfyUI+Manager基线上导入带真实身份的00，确认缺节点提示能发现该包，
经实际Manager界面安装并Restart。Windows4.2.2可能只把requirements/install.py排到下次启动；
必须看重启后实际版本、日志与节点注册，不能把UI排队成功当安装完成。
默认新界面不保证有Fix Dependencies按钮，旧界面名称为Try fix/Reinstall；策略不允许时
使用官方宿主维护流程，不降低安全级别。

frontend1.55.17的普通搜索优先Algolia，空结果不自动转Registry。浏览器Reload可清前端
内存缓存但不能强制外部索引完成；真实cnr_id/ver可帮助工作流直接解析包。已有Manager缓存与
发布后的新用户目录分别记录，不删除旧缓存来伪造新装结果。

安装后用统一配置节点GUI检查环境、找到现有模型候选、后台验证SHA256、原位复用匹配文件，
仅下载缺失的当前所选资产。Original TE包含分片、config/index及tokenizer，不依赖普通
Download all重建目录。错误目标需明确保留旧字节再下载；不擅自覆盖或混用不相关TE分片。
切换系列保留prompt/seed/采样，建议值只经显式按钮应用。

## 5. 当前验收范围与未完成项

- 共享FP8 probe03：Base/SFT × FP8/16-bit × 两次，共8次实际GPU请求；每次512²、2步、解码。4070TiSUPER、cudaMallocAsync，8项CPU测试0skip；263层q/scale指针/hash/bytes不变。不是全步数质量或速度benchmark。
- 此前delta04 Windows137/137项unit通过、0skip，GUI单元32项通过；冻结Turing helper另有Windows8项focused及真实17成员ZIP纯验证通过，不把它写成再次跑完137项或SM75 GPU采样。此前Chromecontext run03通过16项界面检查，没有download/prompt/Manager安装。
- 候选00后续通过19项GUI检查，作业c0f6b40e6dc84e04bc19dc374ac81572全部ready，复用改名子目录W5、TEv2别名与extra-root VAE，模型网络下载0。真实prompt94fd4b29-a191-4ec0-977b-2f01326b2d6d以1024²/16步/默认SFT W5/TE W8A16/seed314159265成功，输出MOSS/00_moss_lowvram_00001_.png；25.76s日志区间不是通用速度承诺。见[当前统一验证](GUIDED_SETUP_VALIDATION.md)。
- 干净基线requirements/install.py与固定Sage安装/fresh native import通过；不等于真实Manager发现→安装→重启→准备闭环，后者仍待完成。
- 历史01–07正常步数HTTP执行、62项测试和原前端回执按原范围保留；见 [PUBLICATION_VALIDATION.md](PUBLICATION_VALIDATION.md) 与docs/validation，不改原数值/SHA。
- SM75仅有历史2080Ti22GB物理卡的W5 auto与独立Turing micro scope；本次新Turing资产不等于新增图像验证。4g/6g是进程预算，RTX50待补、RTX30未承诺全流程、Linux未验收。

最后检查源码包无模型、私有缓存、凭证、环境和内嵌二进制，且保留所有必要tokenizer/vendor/许可。
再记录HF文件提交、GitHub public/Release、Registry实际响应及安装结果。未完成项不能由本地PASS推断完成。

## 官方参考

- [Registry发布](https://docs.comfy.org/registry/publishing)
- [元数据与依赖](https://docs.comfy.org/registry/specifications)
- [Manager安装阶段](https://docs.comfy.org/custom-nodes/backend/manager)
- [安全规范：禁止运行期依赖安装](https://docs.comfy.org/registry/standards)
- [模板分享与模型metadata](https://docs.comfy.org/interface/features/template)
