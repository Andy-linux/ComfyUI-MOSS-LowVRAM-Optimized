# MOSS LowVRAM Optimized 发布顺序与官方机制

## 当前身份与范围

- 主展示名称：**MOSS LowVRAM Optimized**，明确使用KohakuMoss推理组件。
- 正式Registry包ID计划为`kohaku-moss-lowvram-optimized`；首次创建前确定，
  不把可修改的DisplayName与不可随意更换的包ID混淆。
- GitHub目标沿用现有
  [Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized](https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized)。
  目标仍为private；准备与验收已完成。源码上传进度以操作者真实推送回执为准，
  本文不推断已推送或已转为public。
- Registry Publisher为`xianbei`，公开状态已确认为**Active**。本地加密凭证读取/
  解密检查成功；PAT在Registry服务上的认证仍未验证。
  当前没有适用于这项PAT认证的官方只读验证端点，不借发布/写接口试探认证。
  本地成功读取凭证与服务端授权是两件事，不能据此前者宣称发布完成。
- 三个自制FP8文件的HF目标repo尚未由用户选择。不得擅自放入已有W5/W6仓库。
- 本文记录发布顺序；Registry发布及三个FP8的上传仍待确定最终目标。

现有代码许可继续保留KohakuMoss全文及独立组件NOTICE，不修改许可文本。
正式包名保留Kohaku，展示说明也保留组件归属；模型权重许可与代码许可分开记录。

## 1. 先固定身份与安装边界

按官方[Registry发布说明](https://docs.comfy.org/registry/publishing)和
[pyproject规范](https://docs.comfy.org/registry/specifications)，填写真实Repository、
PublisherId、SemVer及依赖声明。发布前只读查询：

```text
GET https://api.comfy.org/nodes/kohaku-moss-lowvram-optimized
GET https://api.comfy.org/publishers/xianbei
GET https://api.comfy.org/publishers/xianbei/nodes
```

Publisher候选可用性端点是
`GET https://api.comfy.org/publishers/validate?username=<候选值>`。
节点404仅说明查询时没有对应可见记录，不代表已预留名称或可以绕过授权。
实际身份、凭证和目标仓库均由操作者确认。

普通依赖在requirements与pyproject声明，Manager负责安装阶段处理；如增加
`install.py`，应使用当前Python、按平台幂等执行，不覆盖全局Torch/CUDA。
当前包要求Python >=3.12；普通依赖已包含`accelerate`。安装声明与实际Python
3.12验收分开，不能把未来所有Python/Torch组合都写成已测试。
官方[安全规范](https://docs.comfy.org/registry/standards)禁止运行期subprocess pip。
可选Flash/Sage安装保持显式；`auto`不依赖未公开的Turing bundle。

## 2. 上游权重直接索引

已有公开权重通过固定revision、精确bytes和SHA256索引，不复制上传：

- Base/SFT INT8、TE和VAE：继续使用`model_manifest.json`里的真实上游来源。
- W5/W6：直接使用
  [Andylinux/MOSS-Image-SFT-W5A8-W6A8](https://huggingface.co/Andylinux/MOSS-Image-SFT-W5A8-W6A8)，
  固定revision `bfb77cb73ae7acb4f788d50d4043c2d71989b0a7`。
  两个公开文件SHA与已测本机产物匹配；保留原README、LOWBIT_MANIFEST和Apache许可引用。
- 原始TE多文件目录由显式下载器完整准备；不重打包上游模型来冒充单文件自动安装。

仅自制FP8或确有必要且允许分发的自编译产物考虑新上传。自制FP8的三个文件、
source/output hash、未训练声明及row/weight-only区别已在`docs/huggingface/`准备。
HF目标未定前保持local-only；不存在的下载URL不写进工作流。

没有LICENSE的历史Turing自编bundle不公开分发。需要的预编译产物优先引用上游
真实release；自行发布前必须有许可、构建来源、平台/ABI版本和SHA256，不能仅凭
“本机能用”推定可分发。

## 3. 先发布真实node pack，再导出missing-node信息

先完成源码与Registry发布并确认可安装，再由真实安装版本导出模板，携带真实
`properties.cnr_id`及`ver`。发布前不伪造`cnr_id`，也不把GitHub仓库名直接当作
Registry ID。保留现有node class IDs以兼容早期PNG/JSON。

官方frontend优先读`cnr_id`，其次aux信息；无身份时可按node name查询Registry，
但Registry提取/索引失败时仍可能无法推断。Missing Nodes是安装引导，不是自动
修复所有Python环境冲突的保证。

## 4. 对齐模型metadata与真实widget

官方[模板说明](https://docs.comfy.org/interface/features/template)要求模型name、
直接下载url、directory匹配；使用固定revision及hash更便于复核。
实际frontend会按widget字符串选择metadata：

- 单文件DiT/TE/VAE使用明确推荐文件名，`properties.models[].name`与之完全一致。
- `auto`适合已有本机映射，但不能依靠它触发某个推荐文件的Missing Models提示。
  当前01已从auto改为精确公开SFT fused文件，且本轮已用该文件16步执行通过；
  它有自己的当前回执，不借用旧auto图PASS。
- Original TE是目录；config/index/tokenizer等JSON/Jinja不在普通模型下载后缀
  白名单中。04已选择`MOSS-Image-Base/text_encoder`，metadata仅列DiT/VAE，
  新目录selector图本轮50步已通过。完整目录用`--assets te_original`显式准备，
  不承诺Download all搞定。
- Desktop可通过宿主bridge写入模型目录；自托管或远程浏览器通常触发浏览器下载，
  还要在真正运行ComfyUI的服务器放置或执行下载器。

一手实现参考：

- [模型metadata选择](https://github.com/Comfy-Org/ComfyUI_frontend/blob/d7251420b82f4ff52c60460350e563fa262186c5/src/platform/workflow/core/utils/modelRequirements.ts)
- [下载后缀与宿主分支](https://github.com/Comfy-Org/ComfyUI_frontend/blob/d7251420b82f4ff52c60460350e563fa262186c5/src/platform/workflowErrors/missingModel/missingModelDownload.ts)
- [Registry包推断](https://github.com/Comfy-Org/ComfyUI_frontend/blob/d7251420b82f4ff52c60460350e563fa262186c5/src/workbench/extensions/manager/composables/nodePack/useWorkflowPacks.ts)
- [Manager安装阶段](https://docs.comfy.org/custom-nodes/backend/manager)

分享JSON/PNG携带图、参数和下载来源，不携带模型、节点源码、Python、CUDA或驱动。

## 5. 分层验收后再扩大支持口径

当前有7组UI/API图；新增06 W5/4g与07 W6/6g采用16步、FP16-fixed、显式W8A16 TE、
FP32 VAE、attention auto和固定seed314159265，均已本轮正常16步通过。七份精确
UI/API图全部HTTP执行成功，UI随请求嵌入PNG。另有实际Windows/Chrome **v4最终PASS**：
最终Note版06导入9节点、20个原输入持久化widget及两个临时widget，状态settled；
真实按钮callback刷新并打开报告，显示4070 Ti SUPER与cudaMallocAsync；六项dialog
检查全部true，点击关闭后dialog移除，序列化20个输入值不变。无console errors、
无`POST /prompt`。最终label/displayName均为“环境状态：依赖检查通过”，2×截图已
确认可见，全部v3检查及新label检查通过。v3数据/dialog通过但disabled文本value被
前端设计隐藏；仅web显示改用官方支持label，readonly/name/value、20输入及06 UI SHA
不变，无GPU/runtime改动，不重算62测试与7图。
最终回执见[frontend-final-20261025.json](validation/frontend-final-20261025.json)，
[v3回执](validation/frontend-dialog-20261025.json)保留原按钮验证及显示问题证据。
只覆盖06，其余六图未逐一浏览器认证，与七图GPU执行独立，也没有新增图像画质认证。
当前Windows 62/62测试通过，官方CLI1.23.0通过，E702/S102/S307零发现；14个已有
公开文件加新下载SFT共15文件44,190,095,264字节完成SHA核验。32文件AST不变证明
连接格式化前后证据；01与06有格式化后运行。完整脱敏回执见
[PUBLICATION_VALIDATION.md](PUBLICATION_VALIDATION.md)。旧5图、41项组件与17个
短执行仍保持原历史口径，不混入当前计数。

七UI模板在执行后只将Note的说明文字改为最终用户说明。原publication测试记录中的
UI SHA不变；`validation/workflow-note-lineage.json`记录前后UI SHA和不变API SHA，
其余节点ID、链接、设置、模型metadata完全不变。不能将后来的UI SHA冒充原实跑SHA。

当前40系SM89有完整release验收。20系仅保留2080Ti **22 GB物理卡**、W5 auto、4g/6g/
unlimited的历史端到端证据，当前版本没有实卡复测；50系待补，Linux未测试。
其中4g/6g是22 GB卡上的软件预算，不是物理4/6GB卡。完整分代表格及脱敏历史摘要
见`ENVIRONMENT_SUPPORT.md`，不把进程显存预算当作物理小卡认证。

发布前验证两种独立体验：缺节点时Registry/Manager安装引导；缺模型时具体文件
提示、下载与hash核验。Original TE另测完整目录准备。最终仍分别记录正常步数
工作流、dtype、模型hash、allocator和可选backend；不从某个格式的PASS推导全部
模式、硬件和平台都通过。

最后核对公开源码包不含模型、缓存、环境、凭证和未授权二进制。Registry打包可
使用`.comfyignore`排除开发素材，但不能误删tokenizer、运行vendor文件和许可证。
获得真实发布结果后，才回填HF URL/revision与Registry身份，重新导出最终JSON/PNG。

## 安装迁移与当前外部发布状态

六个内部class IDs保持不变。不要同时启用旧`moss-optimized`/`ComfyUI-KohakuMOSS`
与新Registry包，以免重复注册；更新或迁移唯一安装。既有本地`moss-optimized`
目录可为启动器兼容保留。私人`models.local.json`按需要备份/迁移，不进入发布包。

准备内容和本轮执行验收已完成；源码上传结果由操作者取得真实回执后记录。当前
这份验证记录不声称已经推送，Registry仍未发布，repo是否转public及三个FP8的HF
目的仓库仍需最终选择。相同既有Windows环境的隔离目录测试，不是新装依赖认证。
