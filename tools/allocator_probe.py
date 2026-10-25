"""Standalone, bounded CUDA/host lifecycle probe. Run in a disposable process."""
import argparse,json,os,sys

def allocator_env(env,backend):
    env=dict(env)
    # Two aliases must agree; preserve non-backend caller options.
    parts=[]
    def options(value):
        depth=0;start=0
        for index,char in enumerate(value):
            if char=='[':depth+=1
            elif char==']':depth-=1
            elif char==',' and depth==0:
                yield value[start:index];start=index+1
        yield value[start:]
    for key in ('PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):
        for item in options(env.get(key,'')):
            item=item.strip()
            if item and not item.startswith('backend:') and item not in parts:parts.append(item)
    # Native-only tuning must not obscure what the async probe actually tests.
    if backend=='cudaMallocAsync':
        parts=[p for p in parts if not p.startswith(('max_split_size_mb:','roundup_power2_divisions:','garbage_collection_threshold:','expandable_segments:'))]
    value=','.join(parts+['backend:'+backend])
    env['PYTORCH_CUDA_ALLOC_CONF']=value;env['PYTORCH_ALLOC_CONF']=value
    return env

def probe(backend):
    os.environ.update(allocator_env(os.environ,backend))
    result={'requested_backend':backend,'status':'RUNNING','checks':{},'scope':'isolated process; no model loaded'}
    try:
        import torch
        torch.cuda.init()
        actual=torch.cuda.memory.get_allocator_backend()
        result.update(backend=actual,torch_version=torch.__version__,cuda_version=torch.version.cuda)
        boot=torch.empty(1,dtype=torch.uint8,device='cuda');torch.cuda.synchronize();del boot
        if actual!=backend:raise RuntimeError('Requested allocator backend was not selected')
        for required in ('get_per_process_memory_fraction','set_per_process_memory_fraction','reset_peak_memory_stats'):
            if not callable(getattr(torch.cuda,required,None)):raise RuntimeError('Required API unavailable: '+required)
    except Exception as error:
        return {**result,'status':'UNSUPPORTED','error':repr(error)}
    try:
        device=0
        def stats():
            return {'allocated_current':torch.cuda.memory_allocated(device),'reserved_current':torch.cuda.memory_reserved(device),
                    'allocated_peak':torch.cuda.max_memory_allocated(device),'reserved_peak':torch.cuda.max_memory_reserved(device)}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
        result['checks']['baseline']=stats()
        x=torch.ones(4*1024*1024,dtype=torch.uint8,device=device)
        torch.cuda.synchronize();result['checks']['allocated']=stats()
        torch.cuda.reset_peak_memory_stats(device);result['checks']['reset_while_live']=stats()
        assert result['checks']['allocated']['allocated_current']>=x.numel()
        assert result['checks']['reset_while_live']['allocated_peak']>=result['checks']['reset_while_live']['allocated_current']
        del x
        torch.cuda.synchronize();result['checks']['after_free']=stats()
        torch.cuda.empty_cache();result['checks']['after_empty_cache']=stats()
        # Hard cap test: intentionally tiny, never reserve a device-sized tensor.
        raw_old=torch.cuda.get_per_process_memory_fraction(device)
        old=min(1.0,max(0.0,float(raw_old)))
        total=torch.cuda.get_device_properties(device).total_memory
        limit=32*1024**2
        rejected=False
        try:
            torch.cuda.set_per_process_memory_fraction(float(limit/total),device)
            try:
                too_big=torch.empty(64*1024**2,dtype=torch.uint8,device=device)
                del too_big
            except torch.OutOfMemoryError:rejected=True
            assert rejected,'Fraction oversize allocation was not rejected'
            small=torch.ones(1024,dtype=torch.uint8,device=device)
            assert int(small.sum().item())==1024
            del small
        finally:
            torch.cuda.set_per_process_memory_fraction(old,device)
        result['checks']['fraction']={'limit_bytes':limit,'oversize_bytes':64*1024**2,'original_fraction_raw':raw_old,'restored_fraction_target':old,'rejected':rejected,'recovered':True,'restored':torch.cuda.get_per_process_memory_fraction(device)==old}
        # Host buffer remains live until the copying stream signals completion.
        host=torch.arange(1024*1024,dtype=torch.int32).pin_memory()
        stream=torch.cuda.Stream();done=torch.cuda.Event()
        with torch.cuda.stream(stream):
            copied=host.to('cuda',non_blocking=True)
            done.record(stream)
        done.synchronize()
        assert torch.equal(copied.cpu(),host)
        result['checks']['pinned_h2d']={'bytes':host.numel()*host.element_size(),'event_completed':done.query(),'equal':True,'host_retained_until_event':True}
        del copied,host
        torch.cuda.synchronize();torch.cuda.empty_cache()
        host_stats=getattr(torch.cuda.memory,'host_memory_stats',None)
        release=getattr(getattr(torch,'accelerator',None),'empty_host_cache',None) or getattr(torch._C,'_host_emptyCache',None)
        cleanup={'supported':release is not None}
        if host_stats:cleanup['before']=host_stats()
        if release:release()
        if host_stats:cleanup['after']=host_stats()
        result['checks']['idle_host_cache']=cleanup
        result['status']='PASS'
    except Exception as error:
        result.update(status='FAILED',error=repr(error))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cudaMallocAsync','native'],default='cudaMallocAsync');a=ap.parse_args()
    result=probe(a.backend);print(json.dumps(result));return 0 if result['status']=='PASS' else 2
if __name__=='__main__':sys.exit(main())
