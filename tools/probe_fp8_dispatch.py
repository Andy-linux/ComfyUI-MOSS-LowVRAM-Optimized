"""Probe selected real FP8 weights, bound functions and measured operator dispatch.

Standalone diagnostic, no Session/global monkeypatch. Reads vendor decision caches
without changing them. A bound 'fp8' op alone is never treated as FP8 GEMM proof.
"""
import argparse
from collections import Counter
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys


def function_info(fn, depth=0):
    if fn is None or depth > 4:
        return None
    if isinstance(fn,functools.partial):
        return {'kind':'partial','function':function_info(fn.func,depth+1),
                'arguments':[str(a) if isinstance(a,(str,int,float,bool,type(None))) else type(a).__name__ for a in fn.args]}
    result={'module':getattr(fn,'__module__',None),
            'qualname':getattr(fn,'__qualname__',type(fn).__name__)}
    try:
        result['source_file']=inspect.getsourcefile(fn)
        result['source_line']=inspect.getsourcelines(fn)[1]
        result['source_sha256']=hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    except (OSError,TypeError):
        result['source_available']=False
    children={}
    for name,cell in zip(getattr(getattr(fn,'__code__',None),'co_freevars',()),getattr(fn,'__closure__',()) or ()):
        try:value=cell.cell_contents
        except ValueError:continue
        if callable(value):children[name]=function_info(value,depth+1)
        elif isinstance(value,(str,int,float,bool,type(None))):children[name]=value
        elif isinstance(value,dict) and all(isinstance(v,(str,int,float,bool,type(None))) for v in value.values()):children[name]=value
    if children:result['closure']=children
    return result


def selected_dispatch(x,base,bias):
    """Read the exact pinned selector key; unavailable/untuned remains explicit."""
    try:
        from kohakumoss_kernels.backends.triton.fp8 import linear
        from kohakumoss_kernels.tuning import tuner
        from kohakumoss_kernels.tuning.buckets import pow2
        n,k=base.q.shape;m=x.reshape(-1,x.shape[-1]).shape[0]
        key=('fwd',x.device.index,pow2(m,16),n,k,x.dtype,base.recipe,bias is None,True)
        run=linear._RESOLVED.get(key)
        table_key=linear._table_key('fwd',m,n,k,x,base.recipe,bias,True)
        cfg=tuner._CHOSEN.get((str(x.device),'fp8_fwd',table_key))
        # A config can exist before a runner is cached. Conversely, a stale
        # selector record cannot prove the bound op actually executed it.
        selection_class=('selected_w16_fallback' if cfg and cfg.get('gemm')=='w16'
                         else 'selected_fp8_candidate_not_hardware_verified' if cfg and run is not None
                         else 'unknown')
        return {'selector_key':str(key),'table_key':table_key,
                'selected_config':cfg,'resolved_runner':function_info(run),
                'selection_settled':run is not None and cfg is not None,
                'selection_class':selection_class,
                'evidence':'read-only pinned vendor _CHOSEN/_RESOLVED; no forced config'}
    except (ImportError,AttributeError,KeyError) as exc:
        return {'selected_config':None,'selection_settled':False,'unavailable_reason':str(exc)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--layers',default='layers.0.proj.q,layers.0.proj.ffn_up_gate,layers.0.proj.ffn_down')
    parser.add_argument('--tokens',default='64,4096',help='actual M rows, comma separated')
    parser.add_argument('--dtype',choices=('bf16','fp16'),default='bf16')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--backend',choices=('auto','eager'),default='auto')
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    tokens=[int(n) for n in args.tokens.split(',')]
    if args.repeats<1 or args.warmup<0 or any(n<1 for n in tokens):parser.error('positive tokens/repeats and nonnegative warmup required')
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime.checkpoints import bootstrap_vendor,inspect_checkpoint
    from runtime.fp8_support import bind_fp8_inference
    info=inspect_checkpoint(args.weights);bootstrap_vendor()
    import torch
    from kohakumoss.model.io.source import CheckpointSource,ModelKeyView
    from kohakumoss.model.io.load import file_layers
    from kohakumoss.quant.compute import ComputeWeight
    from kohakumoss.quant.comfy_format import layer_base
    dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16
    dev=torch.device(args.device)
    view=ModelKeyView(CheckpointSource(args.weights))
    record=info['fp8_record'];comfy=file_layers(view,record)
    results=[]
    def sync():
        if dev.type=='cuda':torch.cuda.synchronize(dev)
    for name in args.layers.split(','):
        if record:
            if name not in record['layers']:raise ValueError('Not a stored FP8 layer: '+name)
            base=ComputeWeight.from_stored(record['recipe'],view.get(name+'.weight'),view.get(name+'.weight_scale'))
        else:
            if name not in comfy or comfy[name].fmt!='fp8':raise ValueError('Not a Comfy weight-only FP8 layer: '+name)
            base=layer_base(comfy[name],view.get)
        hashes={part:hashlib.sha256(memoryview(t.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
                for part,t in [('q',base.q),('scale',base.scale)]}
        base=base.to(dev)
        holder=torch.nn.Module();holder.base=base
        binding=bind_fp8_inference(holder,dtype=dtype,device=dev,backend=None if args.backend=='auto' else 'eager')
        bias=view.get(name+'.bias').to(device=dev,dtype=dtype) if name+'.bias' in view.keys() else None
        for m in tokens:
            x=torch.randn(m,base.q.shape[1],device=dev,dtype=dtype)
            # Separate diagnostic checks forward-only behavior with grad enabled;
            # measured calls below use the same inference_mode as the runtime.
            with torch.enable_grad():
                y=base.linear(x.detach().requires_grad_(True),bias)
                grad={'grad_enabled_during_call':True,'output_requires_grad':y.requires_grad,
                      'output_grad_fn':None if y.grad_fn is None else type(y.grad_fn).__name__}
            del y
            counters=Counter();snapshots=[]
            with torch.inference_mode():
                for _ in range(args.warmup):base.linear(x,bias)
                sync()
                activities=[torch.profiler.ProfilerActivity.CPU]
                if dev.type=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities,record_shapes=True) as prof:
                    for _ in range(args.repeats):
                        with torch.profiler.record_function('moss_fp8_probe/'+name):
                            y=base.linear(x,bias)
                        snap=(selected_dispatch(x,base,bias) if isinstance(base,ComputeWeight) and args.backend!='eager'
                              else {'selected_config':None,'selection_settled':False,'selection_class':'reference_or_weight_only',
                                    'evidence':'explicit eager or weight-only path; FP8 compute selector not consulted'})
                        cfg=snap.get('selected_config')
                        counters[json.dumps(cfg,sort_keys=True) if cfg else 'UNRESOLVED_OR_NOT_APPLICABLE']+=1
                        snapshots.append(snap)
                    sync()
                operators=[{'name':event.key,'count':event.count,
                            'cpu_time_total_us':event.cpu_time_total,
                            'device_time_total_us':getattr(event,'device_time_total',None)} for event in prof.key_averages()]
                kernels=Counter(event.name for event in prof.events() if 'CUDA' in str(event.device_type))
            results.append({'layer':name,'M':m,'N':base.q.shape[0],'K':base.q.shape[1],
                            'source_tensor_sha256':hashes,'binding':binding,
                            'bound_function':function_info(base._linear),'grad_probe':grad,
                            'training_parts_present':[n for n in ('q_t','scale_t','aux_t') if getattr(base,n,None) is not None],
                            'measured_calls':args.repeats,'selected_config_call_counts':dict(counters),
                            'counter_semantics':'selector observations after measured calls; not hardware instruction counters',
                            'dispatch_snapshots':snapshots,'operators':operators,'cuda_kernel_counts':dict(kernels)})
            del x,y
        del holder,base
    report={'weights':info,'device':str(dev),'torch_version':torch.__version__,
            'cuda_version':torch.version.cuda,'results':results,
            'scope':'isolated real-weight linears, not end-to-end sampling or a throughput benchmark',
            'interpretation':'w16 selected_config means dequantized 16-bit fallback. fp8_linear_infer name alone proves no GEMM dtype. Unresolved configs are unknown, not FP8 PASS. Profiler names are corroborating evidence.',
            'global_monkeypatch':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(output.resolve()),'cases':len(results),'status':'PROBED_NOT_QUALITY_QUALIFIED'}))


if __name__=='__main__':main()
