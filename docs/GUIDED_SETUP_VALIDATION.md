# 统一00工作流验证：候选环境、模型复用与安装边界

本页记录新增00工作流的实际验证，和历史01–07回执分别保存。
下列结果不等于新机器从公开Registry完成缺节点安装，亦不表示模型首次网络下载已经测试。
GitHub源码与Turing Release已公开，Registry尚未发布；公共HF目标已建立，但两份row权重仍上传中，尚无远端两权重完成记录。

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

## 独立证据层次

| 验证层 | 已观察结果 | 没有由此证明的内容 |
| --- | --- | --- |
| 干净Python依赖准备 | Windows Python3.12.13 / Torch2.10+cu130 / Manager4.2.2，起初无Triton/accelerate，requirements与install.py退出0；固定Sage隔离准备及fresh native import通过，Torch/Triton版本未替换 | 不是从Manager UI发现缺失节点后安装的记录，不是新装OS认证 |
| 新00界面 | 独立Chrome context run03先通过16项联动/九文件Original TE/metadata/24值序列化检查；候选服务后续19项GUI检查及三模型复用ready | 前一run03没有download/prompt/Manager安装；后续复用场景也没有模型网络下载 |
| 候选服务端到端 | 上述1024²、16步真实prompt成功并输出PNG | 不等于空Manager基线的节点发现、源码下载、依赖安装和重启闭环 |
| 共享FP8 | Base/SFT × FP8/16-bit × 两次，共8请求，512²/2步/解码；263层q/scale指针、hash、bytes不变，8项CPU测试0skip；4070TiSUPER、cudaMallocAsync | 不是全步数画质或普遍速度benchmark，也不覆盖SM75 |
| 单元与冻结helper | 此前delta04 Windows137/137通过、0skip，GUI相关32项通过；冻结Turing helper独立8项focused通过，真实17成员ZIP纯校验通过 | focused不是再次跑完137项；ZIP检查不运行SM75 GPU工作负载 |
| 真正空基线Manager安装 | 独立环境已准备，公开包安装闭环仍待完成 | 不把预先放置候选插件、已有模型或直接执行install.py写成Manager新装成功 |

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
