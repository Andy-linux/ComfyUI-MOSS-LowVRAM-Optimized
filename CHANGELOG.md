# 更新记录

## 0.1.1

- 支持Windows SM89上的普通CPython3.12/3.13、Torch2.10+cu130和固定Triton3.6依赖准备。
- 改进配置检查、Manager入口和资源准备窗口交互。
- 完善请求超时提示与配置变更后的状态刷新。
- 精简安装说明、环境支持表和唯一00工作流引导。
- FP8模型下载固定到包含新版模型说明的仓库revision。

## 0.1.0 — 初始发布候选

- 提供统一00工作流，在一个配置节点选择模型、计算模式和显存预算。
- 支持SFT W5/W6、Base/SFT INT8及Kohaku FP8 row；同一row文件可选择FP8或16-bit计算。
- 提供W8A16、W8A8、W4A16/W4A8和Original BF16/FP16文本编码选项。
- 在界面准备所选资源，支持SHA256校验、匹配文件复用、下载重试及完整Original TE目录。
- 通过Manager安装阶段准备匹配依赖和可选注意力组件。
- 优先使用ComfyUI原生cudaMallocAsync，提供独立启动探测与native回退工具。
- 保留现有节点ID、上游来源和组件许可。
