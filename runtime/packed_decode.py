"""Independent little-endian bit-stream decoder for signed W5/W6.

Format specification (not an implementation copied from another decoder):
each output channel has K unsigned B-bit integers; its bit stream is stored
least-significant bit first in the first axis of a UINT8 [ceil(K*B/8), N]
array. Values use B-bit two's complement. Output is INT8 [N,K].
There is no floating conversion, scaling, quantization or training here.
"""
import triton
import triton.language as tl


@triton.jit
def unpack_to_int8(packed, output, N:tl.constexpr, K:tl.constexpr,
                   B:tl.constexpr, BN:tl.constexpr, BK:tl.constexpr):
    channel=tl.program_id(0)*BN+tl.arange(0,BN)
    column=tl.program_id(1)*BK+tl.arange(0,BK)
    bit=column*B
    byte=bit//8
    shift=bit%8
    valid=(channel[:,None]<N)&(column[None,:]<K)
    address=byte[None,:]*N+channel[:,None]
    lo=tl.load(packed+address,mask=valid,other=0).to(tl.int32)
    hi=tl.load(packed+address+N,
               mask=valid&((byte[None,:]+1)<((K*B+7)//8)),other=0).to(tl.int32)
    unsigned=((lo|(hi<<8))>>shift[None,:])&((1<<B)-1)
    signed=(unsigned^(1<<(B-1)))-(1<<(B-1))
    tl.store(output+channel[:,None]*K+column[None,:],signed.to(tl.int8),mask=valid)
