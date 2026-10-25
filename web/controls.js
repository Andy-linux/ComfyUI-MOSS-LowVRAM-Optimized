import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Resolve -1 on a submitted snapshot. The editable graph keeps -1; saved PNGs
// and queue history contain a reproducible, JavaScript-safe integer.
function freshSeed() {
    const words = new Uint32Array(2);
    crypto.getRandomValues(words);
    return (words[0] & 0x1fffff) * 0x100000000 + words[1];
}

function submittedSeeds(prompt) {
    const randomNodes = Object.entries(prompt?.output ?? {}).filter(
        ([, n]) => n.class_type === "MOSSOptimizedSampler" && n.inputs?.seed === -1
    );
    if (!randomNodes.length) return prompt;
    const result = { ...prompt, output: { ...prompt.output },
        workflow: prompt.workflow ? structuredClone(prompt.workflow) : prompt.workflow };
    for (const [id, original] of randomNodes) {
        const seed = freshSeed();
        result.output[id] = { ...original, inputs: { ...original.inputs, seed } };
        const node = result.workflow?.nodes?.find(n => String(n.id) === String(id));
        if (node?.widgets_values?.length) {
            node.widgets_values[0] = seed;
            if (["fixed", "increment", "decrement", "randomize"].includes(node.widgets_values[1])) {
                node.widgets_values[1] = "fixed";
            }
        }
        const named = node?.widgets_values_named;
        if (named && Object.hasOwn(named, "seed")) {
            named.seed = seed;
            if (Object.hasOwn(named, "control_after_generate")) named.control_after_generate = "fixed";
        }
    }
    return result;
}

const originalQueuePrompt = api.queuePrompt;
api.queuePrompt = function (number, prompt, ...options) {
    return originalQueuePrompt.call(this, number, submittedSeeds(prompt), ...options);
};

function statusWidget(node, name, value) {
    const widget = node.addWidget("text", name, value, () => {}, { serialize: false });
    widget.disabled = true;
    return widget;
}

function setStatus(node, message) {
    if (node._mossStatusWidget) node._mossStatusWidget.value = String(message);
    app.graph?.setDirtyCanvas(true, true);
}

function transientDisplay(nodeType, persistentCount, names, field, initial) {
    // Some frontend versions serialize custom display widgets even when their
    // options say serialize:false. Do not restore old execution status from PNGs.
    const serialize = nodeType.prototype.serialize;
    if (typeof serialize === "function") {
        nodeType.prototype.serialize = function (...args) {
            const data = serialize.apply(this, args);
            if (Array.isArray(data?.widgets_values)) data.widgets_values = data.widgets_values.slice(0, persistentCount);
            if (data?.widgets_values_named) {
                for (const name of names) delete data.widgets_values_named[name];
            }
            return data;
        };
    }
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
        const result = configured?.apply(this, args);
        if (this[field]) this[field].value = initial;
        return result;
    };
}

async function queueMaintenance(node, action) {
    if (node._mossSubmitting) return;
    node._mossSubmitting = true;
    try {
        const prompt = await app.graphToPrompt();
        const target = prompt.output[String(node.id)];
        if (!target || target.class_type !== "MOSSOptimizedControls") {
            throw new Error("维护节点未出现在待提交图中，请检查连接和节点模式。");
        }
        target.inputs.action = action;
        if (action === "unload") {
            // Only the session is an ancestor of an unload request. Never encode
            // prompts or generate an image just to release the session.
            delete target.inputs.positive;
            delete target.inputs.negative;
            delete target.inputs.latent_image;
        }
        const saved = prompt.workflow?.nodes?.find(n => String(n.id) === String(node.id));
        if (saved?.widgets_values?.length) saved.widgets_values[0] = action;
        if (saved?.widgets_values_named && Object.hasOwn(saved.widgets_values_named, "action")) {
            saved.widgets_values_named.action = action;
        }
        const submitting = (action === "prewarm" ? "预热" : "卸载") + "正在提交，等待执行。";
        setStatus(node, submitting);
        const reply = await api.queuePrompt(0, prompt, {
            partialExecutionTargets: [String(node.id)]
        });
        if (!reply?.prompt_id) throw new Error("服务器没有返回排队编号。");
        // A very fast unload can finish before the HTTP reply arrives.
        if (node._mossStatusWidget?.value === submitting) {
            setStatus(node, (action === "prewarm" ? "预热" : "卸载") + "已排队，等待执行。");
        }
    } catch (error) {
        setStatus(node, "提交失败：" + (error?.message ?? String(error)));
        console.error("[KohakuMOSS maintenance]", error);
    } finally {
        node._mossSubmitting = false;
    }
}

function preserveMinusOne(node) {
    const seed = node.widgets?.find(w => w.name === "seed");
    const control = node.widgets?.find(w => w.name === "control_after_generate");
    if (!seed || !control || control._mossMinusOne) return;
    control._mossMinusOne = true;
    for (const hook of ["beforeQueued", "afterQueued"]) {
        const previous = control[hook];
        if (typeof previous !== "function") continue;
        control[hook] = function (...args) {
            if (Number(seed.value) !== -1) return previous.apply(this, args);
            const saved = control.value;
            control.value = "fixed";
            try { return previous.apply(this, args); }
            finally {
                control.value = saved;
                seed.value = -1;
            }
        };
    }
}

app.registerExtension({
    name: "KohakuMOSS.DailyControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MOSSOptimizedSampler") {
            transientDisplay(nodeType, 8, ["最近实际种子"], "_mossSeedWidget", "尚未生成");
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const result = created?.apply(this, args);
                preserveMinusOne(this);
                this._mossSeedWidget = statusWidget(this, "最近实际种子", "尚未生成");
                return result;
            };
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message, ...args) {
                const result = executed?.call(this, message, ...args);
                if (message?.actual_seed?.length && this._mossSeedWidget) {
                    this._mossSeedWidget.value = String(message.actual_seed[0]);
                    app.graph?.setDirtyCanvas(true, true);
                }
                return result;
            };
        }
        if (nodeData.name === "MOSSOptimizedControls") {
            transientDisplay(nodeType, 1, ["预热当前提示词和尺寸", "卸载本 MOSS 会话", "状态"], "_mossStatusWidget", "待机；按钮通过普通队列执行");
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function (...args) {
                const result = created?.apply(this, args);
                this.addWidget("button", "预热当前提示词和尺寸", null,
                    () => queueMaintenance(this, "prewarm"), { serialize: false });
                this.addWidget("button", "卸载本 MOSS 会话", null,
                    () => queueMaintenance(this, "unload"), { serialize: false });
                this._mossStatusWidget = statusWidget(this, "状态", "待机；按钮通过普通队列执行");
                return result;
            };
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message, ...args) {
                const result = executed?.call(this, message, ...args);
                if (message?.moss_status?.length) setStatus(this, message.moss_status[0]);
                return result;
            };
        }
    }
});

