"""One normal Comfy core patcher with accurate frozen-buffer residency accounting.

No extra model lease and no global MM hook. Packed buffers cannot be safely
partially paged through Comfy's parameter-leaf load list, so partial eviction
conservatively performs the standard full core offload. Session residency checks
detect its moved storage before the next request and rebuild installed owners.
"""
def unique_cuda_storage_bytes(model):
    storage={}
    for tensor in list(model.parameters())+list(model.buffers()):
        if tensor.device.type=="cuda" and tensor.numel():
            item=tensor.untyped_storage()
            storage[(str(tensor.device),item.data_ptr())]=item.nbytes()
    return sum(storage.values())

_CLASS=None

def make_core_patcher(model,*,load_device,offload_device):
    global _CLASS
    if _CLASS is None:
        from comfy.model_patcher import ModelPatcher
        class FrozenCoreModelPatcher(ModelPatcher):
            def model_size(self):
                owner=getattr(self.model,"_moss_payload_owner",None)
                if owner is not None and not owner.closed:
                    return unique_cuda_storage_bytes(self.model)
                return super().model_size()

            def loaded_size(self):
                return unique_cuda_storage_bytes(self.model)

            def unpatch_model(self,device_to=None,unpatch_weights=True):
                owner=getattr(self.model,"_moss_payload_owner",None)
                # Comfy calls this with unpatch_weights=False on first load to
                # restore object patches. In that case no tensor is migrated.
                if owner is not None and unpatch_weights and device_to is not None and str(device_to)!=str(self.load_device):
                    owner.close()
                return super().unpatch_model(device_to,unpatch_weights)

            def partially_load(self,device_to,extra_memory=0,force_patch_weights=False):
                owner=getattr(self.model,"_moss_payload_owner",None)
                if owner is not None and owner.closed:
                    raise RuntimeError("Original-payload model was offloaded; session must rebuild before use")
                return super().partially_load(device_to,extra_memory,force_patch_weights)

            def partially_unload(self,device_to,memory_to_free=0,force_patch_weights=False):
                before=self.loaded_size()
                if not before or memory_to_free<=0:
                    return 0
                # This method changes only this core model and its standard patch
                # metadata. It never mutates MM current_loaded_models or sessions.
                self.unpatch_model(device_to,unpatch_weights=True)
                return max(0,before-self.loaded_size())
        _CLASS=FrozenCoreModelPatcher
    return _CLASS(model,load_device=load_device,offload_device=offload_device)
