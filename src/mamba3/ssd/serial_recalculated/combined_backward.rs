// TODO: Implement the combined backward for the MIMO-first SSD.
//
// The previous implementation used the old Mamba-2 interface (x_bnlhp, dt_discretized_bhnl,
// b_bnlgr, c_bnlgr, ...). With the MIMO-first refactor, the interface is now:
//   v_bnlrhp [b,n,l,R,H,P], da_bnlh [b,n,l,H], b_bnlrhn [b,n,l,R,H,N], c_bnlrhn, ...
//
// Until this is updated, the custom backward is not used.
// backward.rs delegates to standard autodiff through K2-K5.
