import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Setup controls belong only to the unified node.
const TYPE = "MOSSOptimizedConfiguration";
const RESUME_KEY = "moss.setup.resume-workflow";
let setupOffered = false;
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
    const resetRecommended = field => {
        const current = value(node, field);
        if (current === "auto" || Object.values(PINNED_NAMES).includes(current)) setValue(node, field, "auto");
    };
    if (["model_family", "dit_storage"].includes(changed)) resetRecommended("dit_checkpoint");
    if (changed === "te_storage") resetRecommended("te_checkpoint");
    if (changed === "vae_asset") resetRecommended("vae_checkpoint");
}
function serviceError(path, code) {
    if (code === 404 && path.includes("/jobs/")) return "准备任务已失效。请刷新检查，再确认准备。";
    if (code === 404) return "MOSS 准备服务未就绪。请在 Manager 确认已安装 MOSS LowVRAM Optimized，应用更改并重启，然后刷新页面重试。";
    if (code === 401 || code === 403) return `ComfyUI 拒绝了请求（HTTP ${code}）。请检查页面连接和访问权限后重试。`;
    return `请求失败（HTTP ${code}），请重试。若持续失败，请查看 ComfyUI 的 MOSS 启动错误。`;
}
async function request(path, payload, { timeoutMs = 45000 } = {}) {
    const controller = new AbortController();
    let timer, timeoutError;
    const deadline = new Promise((_, reject) => {
        timer = setTimeout(() => {
            timeoutError = new Error("请求超时，请重试检查。若刚重启 ComfyUI，请稍候再试。");
            reject(timeoutError);
            controller.abort();
        }, timeoutMs);
    });
    try {
        return await Promise.race([(async () => {
            const response = await api.fetchApi(path, {
                ...(payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }),
                signal: controller.signal,
            });
            if (response.status === 404 || response.status === 401 || response.status === 403) {
                const error = new Error(serviceError(path, response.status));
                error.mossInstallationMissing = response.status === 404 && !path.includes("/jobs/");
                throw error;
            }
            let data;
            try { data = await response.json(); } catch { throw new Error(response.ok ? "MOSS 返回的内容无法读取。请刷新页面，确认插件安装完成后重试。" : serviceError(path, response.status)); }
            if (!response.ok) throw new Error(typeof data?.error === "string" ? data.error : serviceError(path, response.status));
            if (!data || typeof data !== "object") throw new Error("MOSS 返回的检查结果不完整。请刷新页面后重试。");
            if (data.error) throw new Error(typeof data.error === "string" ? data.error : serviceError(path, response.status));
            return data;
        })(), deadline]);
    } catch (error) {
        if (timeoutError) throw timeoutError;
        if (error instanceof TypeError) throw new Error("无法连接 ComfyUI，请确认服务已启动后重试。");
        throw error;
    } finally { clearTimeout(timer); }
}
function compatibilityWarnings(report) {
    const messages = (report.evaluation?.warnings ?? []).map(issue => issue.message).filter(Boolean);
    if (!report.evaluation?.errors?.length) {
        const preparationErrors = report.dependency_plan?.errors ?? [];
        for (const issue of preparationErrors) if (issue.message) messages.push(`自动依赖准备条件：${issue.message}`);
        if (!preparationErrors.length && report.dependency_plan?.status === "unsupported") messages.push("当前环境不满足自动依赖准备条件。");
    }
    return [...new Set(messages)];
}
function needsManager(report) {
    const errors = report.evaluation?.errors ?? [];
    const hostBlocked = report.dependency_plan?.status === "unsupported" || Boolean(report.dependency_plan?.errors?.length);
    return !hostBlocked && errors.some(issue => (issue.code?.startsWith("dependency_") && issue.code !== "dependency_torch") || issue.code === "sm75_triton");
}
function invalidate(node) {
    node.properties ??= {};
    node.properties.models = [];
    node._mossSetupReport = null;
    status(node, "配置已更改，请检查准备状态");
}
async function inspect(node) {
    const picked = selection(node), key = JSON.stringify(picked);
    if (node._mossInspectPending?.key === key) return node._mossInspectPending.promise;
    const serial = (node._mossInspectSerial ?? 0) + 1;
    node._mossInspectSerial = serial;
    const pending = { key };
    pending.promise = (async () => {
        const report = await request("/moss_lowvram/setup/inspect", { selection: picked, verify: false });
        if (!report.environment || !Array.isArray(report.evaluation?.errors) || !Array.isArray(report.assets?.files)) throw new Error("MOSS 检查结果不完整。请刷新页面并检查插件版本。");
        if (node._mossInspectSerial !== serial || key !== JSON.stringify(selection(node))) return null;
        node._mossSetupReport = report;
        node.properties ??= {};
        node.properties.models = [];
        for (const local of report.local_models ?? []) {
            const w = widget(node, local.field);
            if (w) w.options.values = [...new Set(["auto", ...local.choices, w.value])];
        }
        for (const [field, options] of Object.entries(report.options ?? report.capabilities?.options ?? {})) {
            const w = widget(node, field);
            if (!w || !Array.isArray(options)) continue;
            const enabled = options.filter(o => o.enabled !== false).map(o => o.value);
            w.options.values = enabled.includes(w.value) ? enabled : [w.value, ...enabled];
            w.tooltip = options.find(o => o.value === w.value)?.reason || w.tooltip;
        }
        const errors = report.evaluation.errors;
        const warnings = compatibilityWarnings(report);
        status(node, !report.runtime_ready ? "运行环境待准备，点击下方按钮" : report.assets.ready ? "已就绪，可以运行" : "所选模型未齐；可自行放置或使用可选下载", errors.map(e => e.message).join("\n") || warnings.join("\n") || "打开准备窗口查看环境与模型状态。");
        return report;
    })().finally(() => { if (node._mossInspectPending === pending) node._mossInspectPending = null; });
    node._mossInspectPending = pending;
    return pending.promise;
}
const element = (tag, text) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; return e; };
const bytes = n => Number.isFinite(n) ? `${(n / 1073741824).toFixed(2)} GiB` : "大小未知";
function link(text, url) {
    const a = element("a", text); a.href = url; a.target = "_blank"; a.rel = "noopener noreferrer"; a.style.color = "#8dcfff"; return a;
}
function openManager(node, dialog, message) {
    const visible = e => e.getClientRects().length && getComputedStyle(e).visibility !== "hidden" && getComputedStyle(e).display !== "none" && !e.disabled && e.getAttribute("aria-disabled") !== "true";
    const isManager = e => [e.textContent.trim(), e.getAttribute("aria-label") ?? "", e.getAttribute("title") ?? ""].some(text => /^(?:ComfyUI\s*)?Manager$/i.test(text) || text === "管理扩展功能");
    const buttons = [...document.querySelectorAll('button,[role="button"]')].filter(e => visible(e) && isManager(e));
    if (buttons.length === 1) {
        dialog.close();
        buttons[0].click();
        status(node, "已打开 Manager", "完成 MOSS 节点包安装，按提示应用更改或重启，再重新检查。");
        return;
    }
    message.replaceChildren(element("span", "请从 ComfyUI 工具栏打开“管理扩展功能 / Manager”。尚未安装时，请按官方说明启用。 "), link("启用 Manager", "https://docs.comfy.org/manager/install"), element("span", " · "), link("ComfyUI Desktop", "https://www.comfy.org/download"));
}
function environmentDiagnostic(report) {
    const env = report.environment ?? {}, abi = env.python_abi ?? {};
    const safe = value => typeof value === "string" ? value.replace(/[A-Za-z]:[\\/][^\s]+/g, "[路径]").replace(/[a-z][a-z0-9+.-]*:\/\/[^\s]+/gi, "[地址]").slice(0, 160) : typeof value === "boolean" || typeof value === "number" ? value : null;
    return {
        python: safe(env.python_version), implementation: safe(env.python_implementation),
        abi: Object.fromEntries(["abiflags", "soabi", "cache_tag", "py_gil_disabled", "gil_enabled", "free_threaded"].map(key => [key, safe(abi[key])])),
        torch: safe(env.packages?.torch?.version), cuda: safe(env.cuda_build), cuda_available: Boolean(env.cuda_available),
        gpu: safe(env.device?.name), sm: safe(env.device?.sm), triton: safe(env.packages?.triton?.version),
        errors: [...new Set([...(report.evaluation?.errors ?? []), ...(report.dependency_plan?.errors ?? [])].map(issue => safe(issue.code)).filter(Boolean))],
    };
}
async function copyEnvironmentInfo(report, container) {
    const text = JSON.stringify(environmentDiagnostic(report), null, 2);
    try {
        if (!globalThis.navigator?.clipboard?.writeText) throw new Error("clipboard_unavailable");
        await navigator.clipboard.writeText(text);
        container.replaceChildren(element("p", "环境信息已复制。"));
    } catch {
        const field = element("textarea"); field.value = text; field.readOnly = true; field.rows = 12; field.style.width = "100%";
        container.replaceChildren(element("p", "请选中下方环境信息复制。"), field); field.focus?.(); field.select?.();
    }
}
const fileStatus = value => ({ queued: "等待准备", running: "准备中", complete: "准备完成", failed: "准备失败", cancelled: "已取消", ready: "已就绪", local: "本地文件", missing: "待下载", candidate: "已有文件待校验", unverified: "待校验", mismatch: "需替换", custom: "自选文件", publish_pending: "暂无下载源", error: "需要处理", downloading: "下载中", verifying: "校验中" }[value] ?? value ?? "待检查");
async function showSetup(node) {
    if (node._mossSetupDialog?.open) return;
    clearTimeout(node._mossInspectTimer);
    const dialog = element("dialog");
    node._mossSetupDialog = dialog;
    dialog.style.cssText = "max-width:860px;width:88vw;max-height:85vh;overflow:auto;background:#202329;color:#eee;border:1px solid #777;border-radius:10px;padding:24px;font:14px/1.5 sans-serif";
    const heading = element("h2", "MOSS · 配置与准备"), body = element("div"), progress = element("pre"), actions = element("div");
    progress.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere";
    actions.style.cssText = "display:flex;gap:10px;flex-wrap:wrap;margin-top:18px";
    const refresh = element("button", "重新检查"), copy = element("button", "复制环境信息"), download = element("button", "下载勾选项目"), close = element("button", "关闭");
    const prepare = element("button", "准备运行环境"), restart = element("button", "重启并继续");
    let busy = false, checking = false, closed = false, reviewedSelection = null, reviewedRepairs = [], reviewedReport = null;
    const reviewedDownloads = new Set();
    const diagnostics = element("div");
    dialog.append(heading, body, progress, diagnostics, actions); actions.append(prepare, restart, download, refresh, copy, close);
    for (const button of actions.children) button.style.cssText = "padding:9px 14px;border:1px solid #617080;border-radius:6px;background:#303a46;color:#fff;cursor:pointer;font:inherit";
    prepare.style.background = download.style.background = "#076b9c";
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", () => { closed = true; if (node._mossSetupDialog === dialog) node._mossSetupDialog = null; dialog.remove(); });
    function controls() {
        refresh.disabled = busy || checking;
        copy.disabled = !reviewedReport || checking;
        const waitingRestart = Boolean(reviewedReport?.dependency_preparation?.restart_required);
        const hasOptionalPreparation = Boolean(reviewedReport?.dependency_plan.optional_steps?.length);
        prepare.hidden = !reviewedReport || (reviewedReport.runtime_ready && !hasOptionalPreparation) || waitingRestart || reviewedReport.dependency_plan.status === "unsupported";
        prepare.textContent = reviewedReport?.runtime_ready ? "准备可选加速组件" : "准备运行环境";
        prepare.disabled = busy || checking;
        restart.hidden = !waitingRestart;
        restart.disabled = busy || checking;
        download.hidden = !reviewedReport?.assets.files.some(file => file.url && !file.ready);
        download.disabled = busy || checking || reviewedDownloads.size === 0;
        const selectedBytes = (reviewedReport?.assets.files ?? []).filter(file => reviewedDownloads.has(file.destination)).reduce((sum, file) => sum + (file.size_bytes ?? 0), 0);
        download.textContent = reviewedDownloads.size ? `下载勾选项目（${reviewedDownloads.size} 项 · ${bytes(selectedBytes)}）` : "先勾选需要下载的文件";
    }
    function managerHelp(container) {
        const button = element("button", "打开 Manager"), message = element("p", "在 Manager 完成 MOSS 节点包安装，按提示应用更改或重启，再重新检查。");
        button.onclick = () => openManager(node, dialog, message);
        container.append(button, message, link("Manager 安装说明", "https://docs.comfy.org/manager/install"));
    }
    function environmentDetails(report) {
        const details = element("details"), env = report.environment;
        details.append(element("summary", "环境与可选功能"), element("p", `GPU: ${env.device?.name ?? "未发现 CUDA 设备"} · SM${env.device?.sm ?? "?"} · Python ${env.python_version ?? "?"}${env.python_abi?.free_threaded ? " (free-threaded)" : ""} · Torch ${env.packages?.torch?.version ?? "?"} · CUDA ${env.cuda_build ?? "?"} · Triton ${env.packages?.triton?.version ?? "缺失"}`));
        const policy = report.architecture_policy;
        if (policy) {
            details.append(element("p", `${policy.architecture} · ${policy.execution}`), element("p", policy.note));
            const range = report.dependency_plan?.steps?.find(step => step.id === "triton")?.requirement;
            if (range) details.append(element("p", `将准备：${range}；不替换 Torch 或 ComfyUI。`));
        }
        if (report.evaluation.errors.length) {
            for (const issue of report.dependency_plan?.errors ?? []) details.append(element("p", issue.message));
            for (const step of report.dependency_plan?.steps ?? []) details.append(element("p", step.requirement ?? step.message ?? step.id));
        }
        for (const message of compatibilityWarnings(report)) details.append(element("p", message));
        const optional = report.dependency_plan?.optional_dependencies ?? {};
        for (const [name, state] of Object.entries(optional)) {
            if (state.status === "skipped") continue;
            details.append(element("p", `${name}: ${{ available: "可用", installed: "已安装", missing: "未安装", failed: "准备失败", conflict: "存在冲突", publish_pending: "暂无安装源" }[state.status] ?? state.status ?? "未知"}${state.reason ? ` · ${state.reason}` : ""}`));
        }
        for (const step of report.dependency_plan?.optional_steps ?? []) {
            const row = element("p", `${step.id ?? "可选功能"}: ${step.message ?? "安装阶段准备"}`);
            if (step.url) row.append(element("span", " · "), link("来源", step.url));
            details.append(row);
        }
        const unavailable = element("details"); unavailable.append(element("summary", "不可用选项"));
        let unavailableCount = 0;
        for (const [field, options] of Object.entries(report.options ?? {})) {
            for (const option of Array.isArray(options) ? options : []) if (option.enabled === false) {
                unavailable.append(element("p", `${field} · ${option.label ?? option.value}: ${option.reason || "与当前配置不兼容"}`));
                unavailableCount++;
            }
        }
        if (unavailableCount) details.append(unavailable);
        return details;
    }
    async function render() {
        if (busy || checking || closed) return;
        checking = true; reviewedReport = null; reviewedSelection = null; reviewedRepairs = []; reviewedDownloads.clear(); controls();
        progress.textContent = "正在检查环境与模型…";
        try {
            const report = await inspect(node);
            if (closed) return;
            if (!report) { progress.textContent = "配置已更新，请重新检查。"; return; }
            reviewedReport = report; reviewedSelection = JSON.stringify(selection(node));
            const warnings = compatibilityWarnings(report);
            const waitingRestart = report.dependency_preparation.restart_required;
            body.replaceChildren(element("h3", report.runtime_ready ? "运行环境已就绪" : waitingRestart ? "环境已准备，请重启并继续" : "先准备运行环境"));
            if (waitingRestart) {
                body.append(element("p", "点击“重启并继续”。工作流将被保留，重新连接后自动打开本向导。"));
            } else if (!report.runtime_ready) {
                if (report.dependency_plan.status === "unsupported") {
                    for (const issue of report.dependency_plan.errors) body.append(element("p", issue.message));
                    body.append(link("查看当前环境的安装说明", "https://github.com/Andy-linux/ComfyUI-MOSS-LowVRAM-Optimized/blob/main/docs/INSTALLATION_GUIDE.md"));
                } else {
                    body.append(element("p", "点击“准备运行环境”，自动为当前 ComfyUI 补齐缺少的运行组件；不更换 Python、Torch 或模型配置。"));
                    for (const step of report.dependency_plan.steps) body.append(element("p", step.requirement ?? step.message));
                }
                for (const issue of report.evaluation.errors) body.append(element("p", issue.message));
            }
            body.append(environmentDetails(report), element("h3", "本地模型（与环境准备独立）"));
            body.append(element("p", "已有模型可直接选择并运行，无须使用下载助手或取得校验记录。支持 ComfyUI 标准目录、子文件夹和 extra_model_paths.yaml。文件格式必须与所选存储方式匹配。"));
            for (const local of report.local_models ?? []) {
                const row = element("label");
                row.style.cssText = "display:grid;grid-template-columns:120px 1fr;gap:10px;margin:12px 0;align-items:center";
                row.append(element("span", local.label));
                const choose = element("select"); choose.setAttribute("aria-label", local.label + "文件");
                choose.style.cssText = "min-width:0;padding:8px;background:#303a46;color:#fff;border:1px solid #617080;border-radius:5px;font:inherit";
                for (const name of [...new Set(["auto", ...local.choices, local.selected])]) {
                    const option = element("option", name === "auto" ? "自动查找所选类型的本地模型" : name);
                    option.value = name; choose.append(option);
                }
                choose.value = local.selected;
                choose.onchange = async () => { setValue(node, local.field, choose.value); invalidate(node); await render(); };
                row.append(choose); body.append(row, element("p", local.ready ? `${local.filename}：已找到，可直接加载。` : local.message));
            }
            const plan = report.assets, files = plan.files;
            const downloads = element("details"); downloads.open = !plan.ready;
            downloads.append(element("summary", "可选下载助手 · 按文件勾选，不下载其它组件"));
            downloads.append(element("p", "无需下载整套模型。只勾选需要的文件；也可关闭窗口，自行下载后放入对应模型目录。下载不要求先完成运行环境准备。已有本地模型不会被替换。"));
            for (const file of files.filter(file => file.url && !file.ready)) {
                const row = element("div"); row.style.cssText = "padding:10px 0;border-bottom:1px solid #555;overflow-wrap:anywhere";
                const label = element("label"), check = element("input");
                check.type = "checkbox"; check.value = file.destination;
                check.setAttribute("aria-label", "下载 " + file.name); check.style.marginRight = "10px";
                check.onchange = () => { if (check.checked) reviewedDownloads.add(file.destination); else reviewedDownloads.delete(file.destination); controls(); };
                label.append(check, element("strong", `${file.name ?? file.asset_id} · ${bytes(file.size_bytes)}`)); row.append(label);
                if (file.repair_action) row.append(element("p", "确认后保留旧文件，再下载所选版本。"));
                const info = element("details"); info.append(element("summary", "文件与来源"), element("p", file.destination ?? ""));
                if (file.url) info.append(link("下载来源", file.url));
                if (file.sha256) info.append(element("p", `下载校验 SHA256: ${file.sha256}`));
                row.append(info); downloads.append(row);
            }
            if (!files.some(file => file.url && !file.ready)) downloads.append(element("p", "所选本地模型已存在，或该自选文件没有推荐下载地址；无须下载。"));
            body.append(downloads);
            reviewedRepairs = files.filter(f => f.repair_action === "quarantine_then_download").map(f => f.destination);
            progress.textContent = report.runtime_ready && plan.ready ? "运行环境和所选本地模型已就绪。关闭窗口，点击“运行”。" : report.runtime_ready ? "运行环境已就绪。选择已有模型，或按需下载缺失文件；不必下载其它型号。" : "环境与模型分别准备；模型下载可独立使用，开始生成前需两者就绪。";
        } catch (error) {
            if (!closed) { body.replaceChildren(element("h3", "检查未完成")); if (error.mossInstallationMissing) managerHelp(body); progress.textContent = error.message; status(node, "检查未完成，请重试", error.message); }
        } finally { checking = false; controls(); }
    }
    prepare.onclick = async () => {
        busy = true; controls();
        try {
            await request("/moss_lowvram/setup/dependencies", {});
            while (!closed) {
                const state = await request("/moss_lowvram/setup/dependencies");
                progress.textContent = state.log || "正在为当前 ComfyUI 准备运行环境…";
                if (state.state !== "running") {
                    if (state.state === "failed") throw new Error(state.error || state.result?.errors.map(e => e.message).join("\n") || "环境准备失败，完整日志见上方。可再次点击准备继续。");
                    break;
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            busy = false;
            if (!closed) await render();
        } catch (error) { if (!closed) progress.textContent += `\n${error.message}`; }
        finally { busy = false; controls(); }
    };
    restart.onclick = async () => {
        busy = true; controls();
        sessionStorage.setItem(RESUME_KEY, JSON.stringify(app.graph.serialize()));
        const reload = () => location.reload();
        api.addEventListener("reconnected", reload, { once: true });
        progress.textContent = "正在重启 ComfyUI；重新连接后继续。若浏览器询问是否离开，选择“离开”；当前工作流已保留。";
        try {
            let response = await api.fetchApi("/v2/manager/reboot", { method: "POST", headers: { "Content-Type": "application/json" }, body: "null" });
            if (response.status === 404) response = await api.fetchApi("/manager/reboot");
            if (!response.ok) throw new Error("请关闭并重新启动 ComfyUI，然后刷新此页；已保留当前工作流。");
            progress.textContent = "正在重启 ComfyUI；重新连接后继续。若浏览器询问是否离开，选择“离开”；当前工作流已保留。";
        } catch (error) {
            // A successful process restart can close HTTP before its response.
            if (error instanceof TypeError) return;
            api.removeEventListener("reconnected", reload);
            progress.textContent = error.message; busy = false; controls();
        }
    };
    refresh.onclick = render;
    copy.onclick = () => reviewedReport ? copyEnvironmentInfo(reviewedReport, diagnostics) : undefined;
    download.onclick = async () => {
        if (busy || checking || !reviewedReport) return;
        busy = true; controls();
        try {
            if (reviewedSelection !== JSON.stringify(selection(node))) throw new Error("配置已更改，请重新检查后确认准备。");
            const job = await request("/moss_lowvram/setup/download", { selection: selection(node), repair_files: reviewedRepairs.filter(name => reviewedDownloads.has(name)), selected_files: [...reviewedDownloads] });
            while (!closed) {
                const state = await request(`/moss_lowvram/setup/jobs/${encodeURIComponent(job.job_id)}`, undefined, { timeoutMs: 15000 });
                progress.textContent = `${fileStatus(state.state)}: ${bytes(state.bytes_done)} / ${bytes(state.bytes_total)}\n${(state.files ?? []).map(f => `${f.name ?? f.destination ?? f.asset_id ?? "文件"}: ${fileStatus(f.state ?? f.status)}`).join("\n")}${state.error ? `\n${state.error}` : ""}`;
                if (["ready", "complete", "completed", "done", "success", "failed", "error", "cancelled"].includes(state.state)) {
                    if (state.error || ["failed", "error"].includes(state.state)) throw new Error(state.error || "准备失败，请重新检查后重试。");
                    break;
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            busy = false;
            if (!closed) await render();
        } catch (error) {
            reviewedReport = null; reviewedSelection = null;
            if (!closed) progress.textContent = `${error.message}\n点击“重新检查”后重试。`;
        } finally { busy = false; controls(); }
    };
    document.body.append(dialog); dialog.showModal();
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
    async afterConfigureGraph() {
        const saved = sessionStorage.getItem(RESUME_KEY);
        if (saved) {
            sessionStorage.removeItem(RESUME_KEY);
            await app.loadGraphData(JSON.parse(saved));
            return;
        }
        const node = app.graph._nodes.find(n => n.type === TYPE);
        if (!node || setupOffered) return;
        setupOffered = true;
        try {
            const report = await inspect(node);
            if (report && (!report.runtime_ready || !report.assets.ready || new URL(location.href).searchParams.has("moss_setup"))) await showSetup(node);
        } catch (error) { status(node, "请打开准备向导", error.message); }
    },
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
                    node._mossInspectTimer = setTimeout(() => inspect(node).catch(e => status(node, "检查未完成，请重试", e.message)), 300);
                    return output;
                };
            }
            status(node, "首次运行前，请检查准备状态");
            return result;
        };
        const serialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function (...args) {
            const data = serialize.apply(this, args);
            // Keep setup controls out of saved workflows.
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
            inspect(this).catch(e => status(this, "检查未完成，请重试", e.message));
            return result;
        };
    },
});
