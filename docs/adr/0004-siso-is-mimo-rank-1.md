# SISO is MIMO with rank 1

The official code has separate SISO (Triton) and MIMO (Tilelang) kernels. Here,
SISO is `mimo_rank = 1` of the one MIMO implementation. At `m = 1` the fused
`L·M` axis is `chunk_len`, so each kernel already is its SISO form. This keeps
one implementation to read and to test.

## Consequences

The code branches on `m` only where `m` is a real matmul dimension (the
rotation pairing). Two more branches are flags that change only the speed:
`siso_specialization` and `siso_specialization_decode`. The flags give
identical values and gradients.
