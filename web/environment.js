import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const inputs = ["model", "profile", "text_encoder", "attention", "precision",
    "vae_precision", "te_memory", "resident_blocks"];

function selectedValues(node) {
    return Object.fromEntries((node.widgets ?? [])
        .filter(widget => inputs.includes(widget.name))
        .map(widget => [widget.name, widget.value]));
}

function setStatus(node, message, details = message) {
    const widget = node._mossEnvironmentStatus;
    if (!widget) return;
    widget.value = details;
    // Frontend 1.55.17 BaseWidget intentionally returns an empty _displayValue
    // when disabled. Its public label/displayName still renders. Keep the stable
    // widget name and read-only state; show the short status in that label.
    widget.label = `环境状态：${message}`;
    widget.tooltip = details;
}

function showReport(result) {
    const environment = result.environment;
    const evaluation = result.evaluation;
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "max-width:760px;width:85vw;max-height:80vh;overflow:auto;background:#202329;color:#eee;border:1px solid #777;border-radius:8px;padding:24px";
    const title = document.createElement("h2");
    title.textContent = "MOSS · 环境与安装检查";
    const details = document.createElement("pre");
    details.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.6 monospace";
    const lines = [
        `GPU: ${environment.device?.name ?? "未发现可用 CUDA 设备"}`,
        `Python: ${environment.python_version}`,
        `PyTorch: ${environment.packages?.torch?.version ?? "缺失"} / CUDA: ${environment.cuda_build ?? "不可用"}`,
        `Triton: ${environment.packages?.triton?.version ?? "未确认版本"}`,
        `分配器: ${environment.allocator ?? "未确认"}`,
        "",
        ...evaluation.errors.map(item => `需要处理：${item.message}`),
        ...evaluation.warnings.map(item => `支持范围：${item.message}`),
        ...evaluation.actions.map(item => `建议：${item}`),
        "",
        evaluation.qualification,
        "",
        "此检查不安装包、不下载权重、不运行 GPU 运算。安装后请重启 ComfyUI。",
        "完整说明：节点仓库 docs/ENVIRONMENT_SUPPORT.md。",
    ];
    details.textContent = lines.join("\n");
    const close = document.createElement("button");
    close.textContent = "关闭";
    close.style.cssText = "padding:8px 24px;cursor:pointer";
    close.addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => dialog.remove());
    dialog.append(title, details, close);
    document.body.append(dialog);
    dialog.showModal();
}

async function check(node, showDetails = false) {
    const status = node._mossEnvironmentStatus;
    if (!status) return;
    const revision = node._mossEnvironmentRevision = (node._mossEnvironmentRevision ?? 0) + 1;
    setStatus(node, "正在检查环境…");
    try {
        const response = await api.fetchApi("/moss_lowvram/environment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ selection: selectedValues(node), refresh: showDetails }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}；更新节点后请重启 ComfyUI`);
        const result = await response.json();
        if (node._mossEnvironmentDisposed || revision !== node._mossEnvironmentRevision) return;
        const evaluation = result.evaluation;
        const message = evaluation.errors.length
            ? `需要处理 ${evaluation.errors.length} 项`
            : evaluation.warnings.length
                ? "环境可检查（有待验证项）"
                : "依赖检查通过";
        const details = evaluation.errors.length
            ? `需要处理 ${evaluation.errors.length} 项；点击检查查看安装说明`
            : evaluation.warnings.length
                ? "环境可检查；当前组合有待验证项，点击查看"
                : "依赖检查通过；可运行，结果以实际生成验证为准";
        setStatus(node, message, details);
        if (showDetails) showReport(result);
    } catch (error) {
        if (node._mossEnvironmentDisposed || revision !== node._mossEnvironmentRevision) return;
        setStatus(node, "检查失败", `检查失败：${error.message ?? error}`);
    }
    app.graph?.setDirtyCanvas(true, true);
}

function schedule(node) {
    clearTimeout(node._mossEnvironmentTimer);
    node._mossEnvironmentTimer = setTimeout(() => check(node), 150);
}

app.registerExtension({
    name: "MOSS.LowVRAM.Environment",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MOSSOptimizedSessionLoader") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this._mossEnvironmentStatus = this.addWidget("text", "环境状态", "等待检查", () => {}, { serialize: false });
            this._mossEnvironmentStatus.disabled = true;
            setStatus(this, "等待检查");
            this.addWidget("button", "检查环境与安装说明", null, () => check(this, true), { serialize: false });
            schedule(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            schedule(this);
            return result;
        };
        const changed = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, ...args) {
            const result = changed?.call(this, name, ...args);
            if (inputs.includes(name)) schedule(this);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            this._mossEnvironmentDisposed = true;
            clearTimeout(this._mossEnvironmentTimer);
            return removed?.apply(this, args);
        };
        const serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function (...args) {
            const data = serialize.apply(this, args);
            // Some frontend versions serialize display widgets despite the flag.
            // Keep the existing 20-input contract for old JSON/PNG workflows.
            if (Array.isArray(data.widgets_values)) data.widgets_values = data.widgets_values.slice(0, 20);
            if (data.widgets_values_named) {
                delete data.widgets_values_named["环境状态"];
                delete data.widgets_values_named["检查环境与安装说明"];
            }
            return data;
        };
    },
});
