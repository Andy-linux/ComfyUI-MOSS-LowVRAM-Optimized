"""SM75 FP16 W4 codes reference decoder without torch.compile.

Per-instance bind replacement only. The original integer unpack, float codebook /
group-scale product, round/clamp and INT8 result are unchanged. Linear dispatch
remains the pinned upstream dispatch; this is not an A8 activation conversion.
"""
from types import MethodType

def attach(base):
    from kohakumoss.quant.w4a8 import W4A8Weight
    if type(base) is not W4A8Weight:
        return False
    def bind(self,dtype,device=None):
        import torch
        from functools import partial
        from kohakumoss_kernels import registry
        from kohakumoss.quant.w4a8 import decode_codes,_eager_for,_rotated_kernel,_plain_kernel
        device=self.q.device if device is None else torch.device(device)
        if dtype!=torch.float16 or device.type not in ("cpu","cuda"):
            raise ValueError("Reference W4 decoder binding is FP16 CPU storage / SM75 compute only")
        if device.type=="cuda" and tuple(torch.cuda.get_device_capability(device))!=(7,5):
            raise ValueError("Reference W4 decoder CUDA binding is SM75 only")
        self.decoder=decode_codes
        self._linear=_eager_for(self)
        if device.type=="cpu":
            return
        if not self.decodes_to_int8:
            return
        name="quant_linear_rotated" if self.rotation else "quant_linear"
        query={"group":self.rotation} if self.rotation else {}
        try:
            kernel=registry.resolve(name,device,dtype,fmt="int8",**query)
        except (registry.NoImplementationError,KeyError):
            return
        self._linear=partial(_rotated_kernel if self.rotation else _plain_kernel,kernel)
    base.bind=MethodType(bind,base)
    base._sm75_reference_decode=True
    return True
