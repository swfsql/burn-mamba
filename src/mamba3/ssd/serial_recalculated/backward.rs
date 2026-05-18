// TODO: Implement a memory-efficient custom backward for the MIMO-first SSD.
//
// The previous implementation used the old Mamba-2 interface (x_bnlhp, dt_discretized_bhnl,
// b_bnlgr, c_bnlgr, d_h, ...). With the MIMO-first refactor, the interface is now:
//   v_bnlrhp [b,n,l,R,H,P], da_bnlh [b,n,l,H], b_bnlrhn [b,n,l,R,H,N], c_bnlrhn, ...
//
// Until this is updated, Autodiff<B, C> uses the default trait implementation
// (standard autodiff through K2-K5), which is correct but not memory-optimised.
//
// The custom backward logic lives in combined_backward.rs and also needs updating.
