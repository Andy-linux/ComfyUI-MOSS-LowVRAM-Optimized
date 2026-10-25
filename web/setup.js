import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Only the new entry point is enhanced; historical node widget order is sacred.
const TYPE = "MOSSOptimizedConfiguration";
const INTEGER_OP = "INT8 × INT8 → INT32";
const NEW_FIELDS = ["model_family", "dit_storage", "dit_compute", "te_storage", "te_compute", "profile", "precision", "attention", "vae_precision", "vae_asset"];
const PINNED_NAMES = {
    "SFT/Packed W5": "q5_ridge_gptq.safetensors",
    "SFT/Packed W6": "q6_direct.safetensors",
    "SFT/INT8 ConvRot": "moss_image_sft_int8_convrot.safetensors",
    "Base/INT8 ConvRot": "moss-image-base-dit-int8-convrot.safetensors",
    "SFT/Kohaku FP8 row": "moss_image_sft_kohaku_fp8_row.safetensors",
    "Base/Kohaku FP8 row": "moss_image_base_kohaku_fp8_row.safetensors",
    "Sumomo INT8 ConvRot v2": "moss-image-qwen35-int8-convrot-v2.safetensors",
    "Comfy INT8 ConvRot": "qwen3.5_9b_int8_convrot.safetensors",
    "Comfy W4 GPTQ": "qwen3.5_9b_w4a8_gptq.safetensors",
    "Original float": "MOSS-Image-Base/text_encoder",
    "Original MOSS": "moss-image-vae.safetensors",
    "Flux2 small repack": "flux2-vae_small.safetensors",
};
const widget = (node, name) => node.widgets?.find(w => w.name === name);
const value = (node, name) => widget(node, name)?.value;
function selection(node) {
    return Object.fromEntries((node.widgets ?? []).filter(w => !w._mossTransient).map(w => [w.name, w.value]));
}
function setValue(node, name, next) {
    const w = widget(node, name);
    if (!w) return;
    if (Array.isArray(w.options?.values) && !w.options.values.includes(next)) w.options.values = [...w.options.values, next];
    w.value = next;
}
function status(node, short, detail = short) {
    const w = node._mossSetupStatus;
    if (w) { w.label = `MOSS: ${short}`; w.value = detail; w.tooltip = detail; }
    node.setDirtyCanvas?.(true, true);
}
function transient(w) {
    w._mossTransient = true;
    w.options = { ...(w.options ?? {}), serialize: false };
    return w;
}
function structuralChoices(node, changed) {
    const base = value(node, "model_family") === "Base";
    const stores = base ? ["INT8 ConvRot", "Kohaku FP8 row"] : ["Packed W5", "Packed W6", "INT8 ConvRot", "Kohaku FP8 row"];
    const dw = widget(node, "dit_storage");
    dw.options.values = stores;
    if (!stores.includes(dw.value)) dw.value = stores[0];
    const ops = dw.value === "Kohaku FP8 row" ? ["FP8", "16-bit"] : [INTEGER_OP];
    const cw = widget(node, "dit_compute");
    cw.options.values = ops;
    if (!ops.includes(cw.value)) cw.value = ops[0];
    const teOps = {
        "Sumomo INT8 ConvRot v2": ["W8A16", "W8A8"], "Comfy INT8 ConvRot": ["W8A16", "W8A8"],
        "Comfy W4 GPTQ": ["W4A16 (legacy)", "W4A8"], "Original float": ["FP16", "BF16"],
    }[value(node, "te_storage")];
    const tw = widget(node, "te_compute");
    tw.options.values = teOps;
    if (!teOps.includes(tw.value)) tw.value = teOps[0];
    if (["model_family", "dit_storage"].includes(changed)) setValue(node, "dit_checkpoint", PINNED_NAMES[`${value(node, "model_family")}/${dw.value}`]);
    if (changed === "te_storage") setValue(node, "te_checkpoint", PINNED_NAMES[value(node, "te_storage")]);
    if (changed === "vae_asset") setValue(node, "vae_checkpoint", PINNED_NAMES[value(node, "vae_asset")]);
}
async function request(path, payload) {
    const response = await api.fetchApi(path, payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    let data;
    try { data = await response.json(); } catch { throw new Error(`Setup service returned HTTP ${response.status}; restart ComfyUI after updating this plugin.`); }
    if (!response.ok || data.error) throw new Error(typeof data.error === "string" ? data.error : `HTTP ${response.status}`);
    return data;
}
function invalidate(node) {
    node.properties ??= {};
    node.properties.models = []; // Clear immediately; never retain a previous mode's download link.
    node._mossSetupReport = null;
    status(node, "Selection changed — check preparation");
}
async function inspect(node, refresh = false) {
    const serial = (node._mossInspectSerial ?? 0) + 1;
    node._mossInspectSerial = serial;
    const picked = selection(node);
    const report = await request("/moss_lowvram/setup/inspect", { selection: picked, verify: false });
    if (node._mossInspectSerial !== serial || JSON.stringify(picked) !== JSON.stringify(selection(node))) return null;
    node._mossSetupReport = report;
    node.properties ??= {};
    node.properties.models = Array.isArray(report.models) ? report.models : [];
    for (const [field, options] of Object.entries(report.options ?? report.capabilities?.options ?? {})) {
        const w = widget(node, field);
        if (!w || !Array.isArray(options)) continue;
        const enabled = options.filter(o => o.enabled !== false).map(o => o.value);
        // Keep a selected but hardware-incompatible value visible, so the
        // authoritative error remains understandable; do not silently change it.
        w.options.values = enabled.includes(w.value) ? enabled : [w.value, ...enabled];
        w.tooltip = options.find(o => o.value === w.value)?.reason || w.tooltip;
    }
    const errors = report.evaluation?.errors ?? [];
    status(node, errors.length ? `${errors.length} environment issue(s)` : report.assets?.ready ? "Ready — selected files present" : "Models need preparation", errors.map(e => e.message).join("\n") || report.evaluation?.qualification || "Inspect selected files before queueing");
    return report;
}
const element = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
const bytes = n => Number.isFinite(n) ? `${(n / 1073741824).toFixed(2)} GiB` : "size unknown";
function link(text, url) {
    const a = element("a", text); a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.style.color = "#8dcfff"; return a;
}
function openManager(message) {
    // Invoke an existing visible UI control. No guessed Manager network API,
    // shell installer, security setting, or package mutation is used here.
    const button = [...document.querySelectorAll("button")].find(b => /^(comfyui\s*)?manager$/i.test(b.textContent.trim()) || /comfyui.manager/i.test(b.getAttribute("title") ?? ""));
    const instructions = "在 Manager 完成此节点包安装；若提示需要重启，先重启 ComfyUI 再回来检查。Windows Manager 可能将 requirements 和 install.py 的依赖准备延迟到重启阶段执行，请等待重启后的安装任务完成。安装失败时，按当前 Manager 实际提供的重试或重装入口处理；没有对应入口或宿主不兼容时，请使用官方 ComfyUI 安装维护向导。自动准备仅覆盖下方明确列出的宿主范围；不降低安全级别，也不自动替换 Torch/CUDA。";
    if (button) button.click();
    message.replaceChildren(element("span", button ? instructions : `本页面未找到 Manager 入口。${instructions}`), element("span", " "), link("官方 ComfyUI 安装维护向导", "https://docs.comfy.org/installation/desktop"));
}
async function showSetup(node) {
    const dialog = element("dialog");
    dialog.style.cssText = "max-width:860px;width:88vw;max-height:85vh;overflow:auto;background:#202329;color:#eee;border:1px solid #777;border-radius:10px;padding:24px;font:14px/1.5 sans-serif";
    const heading = element("h2", "MOSS · Prepare selected configuration");
    const body = element("div");
    const progress = element("pre"); progress.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere";
    const actions = element("div"); actions.style.cssText = "display:flex;gap:10px;flex-wrap:wrap;margin-top:18px";
    const refresh = element("button", "Refresh environment and files");
    const download = element("button", "Confirm download missing selected files");
    const close = element("button", "Close");
    let busy = false, closed = false, reviewedSelection = null, reviewedRepairs = [];
    dialog.append(heading, body, progress, actions); actions.append(refresh, download, close);
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", () => { closed = true; dialog.remove(); });
    document.body.append(dialog); dialog.showModal();
    async function render() {
        if (busy) return;
        progress.textContent = "Checking this ComfyUI process and selected model files…";
        download.disabled = true;
        try {
            const report = await inspect(node, true);
            reviewedSelection = report ? JSON.stringify(selection(node)) : null;
            if (!report || closed) return;
            body.replaceChildren();
            const env = report.environment ?? {}, evaluation = report.evaluation ?? {};
            body.append(element("h3", "1 · Environment and Manager"), element("p", `GPU: ${env.device?.name ?? "No CUDA device"} · Python ${env.python_version ?? "unknown"} · torch ${env.packages?.torch?.version ?? "missing"} · Triton ${env.packages?.triton?.version ?? "missing"}`));
            for (const issue of [...(evaluation.errors ?? []), ...(evaluation.warnings ?? [])]) body.append(element("p", `${issue.code}: ${issue.message}`));
            body.append(element("p", evaluation.qualification ?? "Environment qualification unavailable"));
            const manager = element("button", "Open Manager / installation guidance");
            const guidance = element("p", "在 Manager 完成此节点包安装；若提示重启，请先重启再检查。Windows Manager 可能把 requirements / install.py 准备延迟到重启阶段执行。安装失败时使用当前界面实际提供的重试或重装入口；无入口或宿主不兼容时使用官方 ComfyUI 安装维护向导。不降低安全级别，不承诺任意宿主均可自动准备。");
            manager.onclick = () => openManager(guidance);
            body.append(manager, guidance);
            const dependency = report.dependency_plan;
            if (dependency) {
                body.append(element("h4", "安装阶段自动准备范围"), element("p", `自动准备计划：${dependency.status ?? "unknown"}。此状态仅描述 Manager 安装阶段的自动准备范围；unsupported 不等于运行环境不可用。能否运行由上方运行环境 evaluation 的错误决定。此对话框不执行依赖安装。`));
                if (dependency.qualification) body.append(element("p", dependency.qualification));
                for (const step of dependency.steps ?? []) {
                    const row = element("p", `${step.message ?? step.id ?? "准备步骤"}${step.requirement ? ` (${step.requirement})` : ""}`);
                    if (step.url) row.append(element("span", " "), link("固定来源", step.url));
                    body.append(row);
                }
                for (const issue of [...(dependency.errors ?? []), ...(dependency.warnings ?? [])]) body.append(element("p", `${issue.code ?? "准备说明"}: ${issue.message ?? issue}`));
                const optional = dependency.optional_dependencies ?? {};
                if (Object.keys(optional).length || (dependency.optional_steps ?? []).length) {
                    body.append(element("h4", "可选注意力扩展 · 默认 auto 不依赖此项"), element("p", "以下扩展仅在后端列出的固定宿主范围内于安装阶段准备。准备失败、未发布或宿主不兼容时保持 attention=auto；这些扩展不计入默认路径的必需依赖。installed 仅表示固定文件/来源检查，available 也不代表已运行 GPU 验证。"));
                    for (const [name, state] of Object.entries(optional)) {
                        body.append(element("p", `${name}: ${state.status ?? "unknown"}${state.reason ? ` · ${state.reason}` : ""}${state.verification ? ` · ${state.verification}` : ""}${state.scope ? ` · ${state.scope}` : ""}`));
                    }
                    for (const step of dependency.optional_steps ?? []) {
                        const row = element("p", `${step.id ?? "可选扩展"}: ${step.message ?? "安装阶段准备"}`);
                        if (step.url) row.append(element("span", " "), link("固定可选来源", step.url));
                        if (step.sha256) row.append(element("span", ` · SHA256: ${step.sha256}`));
                        body.append(row);
                    }
                }
                body.append(element("p", "自动准备未覆盖或安装未完成时，请通过当前 Manager 实际入口或官方宿主维护流程处理，完成后重启并刷新检查。"));
            }
            const unavailable = Object.entries(report.options ?? {}).flatMap(([field, options]) => Array.isArray(options) ? options.filter(option => option.enabled === false).map(option => ({ field, ...option })) : []);
            if (unavailable.length) {
                const details = element("details");
                details.append(element("summary", `不可用选项及原因 (${unavailable.length})`));
                for (const option of unavailable) details.append(element("p", `${option.field} · ${option.label ?? option.value}: ${option.reason || "与当前所选配置不兼容"}`));
                body.append(details);
            }
            body.append(element("h3", "2 · Selected model plan"));
            const plan = report.assets ?? {}, files = plan.files ?? [];
            body.append(element("p", `Total: ${bytes(plan.total_bytes)} · Missing: ${bytes(plan.missing_bytes)}. Downloads run on the ComfyUI server; sources and hashes are fixed by the plugin manifest. Original float TE includes its complete shard/tokenizer bundle.`));
            for (const file of files) {
                const row = element("div"); row.style.cssText = "padding:8px 0;border-bottom:1px solid #555;overflow-wrap:anywhere";
                row.append(element("strong", `${file.name ?? file.asset_id} · ${file.status ?? "unknown"} · ${bytes(file.size_bytes)}`), element("div", file.destination ?? ""));
                if (file.message) row.append(element("div", file.message));
                if (file.resolved_name) row.append(element("div", `Verified existing model: ${file.resolved_name}`));
                for (const rejected of file.candidate_mismatches ?? []) row.append(element("div", `Existing file retained: ${rejected.name} · ${rejected.reason}`));
                if (file.status === "candidate") row.append(element("div", `${file.candidate_count ?? 0} same-size registered candidate(s). Prepare verifies full SHA256 before reuse; a different filename is not proof of matching weights.`));
                if (file.repair_action === "quarantine_then_download") row.append(element("strong", "Confirming preparation moves this rejected existing file into .moss_lowvram_rejected in the same directory, preserving its bytes, then downloads the pinned replacement. It does not overwrite or delete the original."));
                if (file.url) row.append(link("Pinned source", file.url));
                if (file.sha256) row.append(element("div", `SHA256: ${file.sha256}`));
                body.append(row);
            }
            body.append(element("h3", "3 · Download, restart if repaired, then queue"), element("p", "Confirm downloads below. A failed download can be retried. Refresh after installing dependencies or restarting. Changing family does not alter prompts, seed or steps; use Apply recommended sampling separately."));
            reviewedRepairs = files.filter(f => f.repair_action === "quarantine_then_download").map(f => f.destination);
            const repairs = reviewedRepairs.length > 0;
            download.textContent = repairs ? "Confirm preserve rejected files and download replacements" : "Confirm download / verify selected files";
            download.disabled = Boolean(plan.ready) || files.length === 0;
            progress.textContent = plan.ready ? "Selected files are present. Environment issues above must also be resolved before queueing." : "Review file sizes and source links, then confirm download.";
        } catch (error) { progress.textContent = error.message; status(node, "Preparation check failed", error.message); }
    }
    refresh.onclick = render;
    download.onclick = async () => {
        if (busy) return;
        busy = true; download.disabled = true; refresh.disabled = true;
        try {
            if (reviewedSelection !== JSON.stringify(selection(node))) throw new Error("The configuration changed. Refresh the plan and review its sizes before confirming download.");
            const job = await request("/moss_lowvram/setup/download", { selection: selection(node), repair_files: reviewedRepairs });
            while (!closed) {
                const state = await request(`/moss_lowvram/setup/jobs/${encodeURIComponent(job.job_id)}`);
                progress.textContent = `${state.state}: ${bytes(state.bytes_done)} / ${bytes(state.bytes_total)}\n${(state.files ?? []).map(f => `${f.name ?? f.destination ?? f.asset_id ?? "file"}: ${f.state ?? f.status ?? "pending"}`).join("\n")}${state.error ? `\n${state.error}` : ""}`;
                if (["ready", "complete", "completed", "done", "success", "failed", "error", "cancelled"].includes(state.state)) {
                    if (state.error || ["failed", "error"].includes(state.state)) throw new Error(state.error || "Download failed. Refresh and retry; existing files will be checked.");
                    break;
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            busy = false; refresh.disabled = false;
            if (!closed) await render();
        } catch (error) { progress.textContent = `${error.message}\nUse Refresh, then confirm download to retry.`; }
        finally { busy = false; refresh.disabled = false; download.disabled = false; }
    };
    await render();
}
function applySampling(node) {
    const steps = value(node, "model_family") === "Base" ? 50 : 16;
    // Only connected samplers belong to this configuration; do not alter seeds.
    const targets = new Set();
    for (const output of node.outputs ?? []) for (const id of output.links ?? []) {
        const edge = app.graph.links[id];
        const target = edge && app.graph.getNodeById(edge.target_id);
        if (target?.comfyClass === "MOSSOptimizedSampler" || target?.type === "MOSSOptimizedSampler") targets.add(target);
    }
    for (const target of targets) for (const [name, v] of Object.entries({steps, cfg: 3, sampler_name: "euler", scheduler: "sgm_uniform", shift: 5, slg_layer: 22})) setValue(target, name, v);
    status(node, targets.size ? `Applied ${steps}-step sampling to ${targets.size} connected sampler(s)` : "Connect a MOSS sampler to apply sampling");
}
app.registerExtension({
    name: "MOSS.LowVRAM.UnifiedSetup",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const node = this;
            node._mossSetupStatus = transient(node.addWidget("text", "moss_setup_status", "", () => {}, { serialize: false }));
            node._mossSetupStatus.disabled = true;
            transient(node.addWidget("button", "Prepare / check selected configuration", null, () => showSetup(node), { serialize: false }));
            transient(node.addWidget("button", "Apply recommended sampling parameters", null, () => applySampling(node), { serialize: false }));
            for (const w of node.widgets.filter(w => !w._mossTransient)) {
                const callback = w.callback;
                w.callback = function () {
                    const output = callback?.apply(this, arguments);
                    if (NEW_FIELDS.includes(w.name)) structuralChoices(node, w.name);
                    invalidate(node);
                    clearTimeout(node._mossInspectTimer);
                    node._mossInspectTimer = setTimeout(() => inspect(node).catch(e => status(node, "Check unavailable", e.message)), 300);
                    return output;
                };
            }
            status(node, "Prepare / check before first queue");
            return result;
        };
        const serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function (...args) {
            const data = serialize.apply(this, args);
            // Frontend 1.55 serializes display widgets despite serialize:false.
            // All 24 persistent inputs precede these three setup-only widgets.
            if (Array.isArray(data.widgets_values)) data.widgets_values = data.widgets_values.slice(0, 24);
            if (data.widgets_values_named) {
                delete data.widgets_values_named.moss_setup_status;
                delete data.widgets_values_named["Prepare / check selected configuration"];
                delete data.widgets_values_named["Apply recommended sampling parameters"];
            }
            return data;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            structuralChoices(this);
            invalidate(this);
            inspect(this).catch(e => status(this, "Check unavailable", e.message));
            return result;
        };
    },
});
