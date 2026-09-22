//! TEMPORARY (graph-capture handoff, phase 2): how is a captured graph's input
//! refreshed so that the replay reads it?
//!
//! A replay reads the exact device buffers the capture touched, so a refresh has
//! to land *in* those buffers, never swap the handle. Each experiment prints what
//! it checks and `OK`/`FAIL`; nothing panics, so one run reports every case.
//!
//! Every stable buffer is allocated **before** `capture` (the window pins what it
//! allocates itself, which would make an in-place op on it copy instead).

use std::cell::RefCell;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::capture;

fn main() {
    let device = Device::default();
    println!("device: {device:?}");

    e1_baseline(&device);
    e2_slice_assign_refresh(&device);
    e3_clone_held_refresh(&device);
    #[cfg(feature = "cubecl")]
    e4_client_write(&device);
    e5_self_advancing(&device);
    e6_timing(&device, 256, 300);
}

fn vals<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone().into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4 * (1.0 + y.abs()))
}

fn verdict(name: &str, ok: bool, detail: impl std::fmt::Display) {
    println!("  [{}] {name}: {detail}", if ok { "OK  " } else { "FAIL" });
}

/// The memory id of `t`'s buffer (cubecl only): equal ids ⇔ same device buffer.
#[cfg(feature = "cubecl")]
fn buffer_id<const D: usize, K>(t: &Tensor<D, K>) -> Option<usize>
where
    K: burn::tensor::kind::Basic
        + burn::tensor::BackendPrimitive<burn_cubecl::Cube, Primitive = burn_cubecl::tensor::CubeTensor>,
{
    let cube = t.clone().try_into_primitive::<burn_cubecl::Cube>().ok()?;
    Some(cube.handle.memory.descriptor().id.value)
}
#[cfg(not(feature = "cubecl"))]
fn buffer_id<const D: usize, K: burn::tensor::kind::Basic>(_t: &Tensor<D, K>) -> Option<usize> {
    None
}

/// E1 — the burn test, as a sanity check: replaying a pure closure.
fn e1_baseline(device: &Device) {
    println!("E1 baseline: capture `x*2+1`, replay 3×");
    let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], device);
    let expected = vals(&(x.clone() * 2.0 + 1.0));
    let mut graph = capture(device, || x.clone().mul_scalar(2.0).add_scalar(1.0));
    verdict("is_hardware", true, graph.is_hardware());
    for k in 0..3 {
        // Safety: `x` outlives `graph` (declared first, dropped last); one thread.
        let out = vals(unsafe { graph.replay() });
        verdict(&format!("replay {k}"), close(&out, &expected), format!("{out:?}"));
    }
}

/// E2 — refresh by an in-place `slice_assign` on a **uniquely owned** stable
/// tensor. The closure borrows it through a `RefCell` and clones it only for the
/// duration of a run, so between replays the stable tensor is the handle's only
/// owner and cubecl's `can_mut` lets the kernel write into the same buffer.
fn e2_slice_assign_refresh(device: &Device) {
    println!("E2 refresh via in-place slice_assign (closure holds no clone)");
    let slot = RefCell::new(Some(Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], device)));
    let run = || {
        let x = slot.borrow().as_ref().unwrap().clone();
        x.mul_scalar(2.0).add_scalar(1.0)
    };
    let mut graph = capture(device, run);
    let id0 = buffer_id(slot.borrow().as_ref().unwrap());
    for k in 1..=3 {
        let new = [10.0 * k as f32, 1.0, -2.0, 0.5 * k as f32];
        {
            let mut s = slot.borrow_mut();
            let x = s.take().unwrap();
            *s = Some(x.slice_assign([0..4], Tensor::from_floats(new, device)));
        }
        let id = buffer_id(slot.borrow().as_ref().unwrap());
        // Safety: the stable tensor lives in `slot`, which outlives `graph`, and
        // its buffer is never freed (asserted by the id check); one thread.
        let out = vals(unsafe { graph.replay() });
        let expected: Vec<f32> = new.iter().map(|v| 2.0 * v + 1.0).collect();
        verdict(
            &format!("refresh {k}"),
            close(&out, &expected) && id == id0,
            format!("out {out:?}, buffer {id0:?} → {id:?}"),
        );
    }
}

/// E3 — the hazard: the closure owns a clone, so the stable tensor's handle is
/// shared, `slice_assign` copies, and the refresh swaps the handle. Expected to
/// read stale (the old buffer stays alive in the closure, so this is safe).
fn e3_clone_held_refresh(device: &Device) {
    println!("E3 hazard: closure holds a clone ⇒ slice_assign copies (expect stale)");
    let mut x = Tensor::<1>::from_floats([1.0, 2.0, 3.0, 4.0], device);
    let held = x.clone();
    let mut graph = capture(device, move || held.clone().mul_scalar(2.0).add_scalar(1.0));
    let id0 = buffer_id(&x);
    let new = [5.0, 6.0, 7.0, 8.0];
    x = x.slice_assign([0..4], Tensor::from_floats(new, device));
    let id = buffer_id(&x);
    // Safety: the captured buffer is owned by the closure inside `graph`.
    let out = vals(unsafe { graph.replay() });
    let stale = close(&out, &[3.0, 5.0, 7.0, 9.0]);
    verdict(
        "stale as predicted",
        stale || !graph.is_hardware(),
        format!("out {out:?}, buffer {id0:?} → {id:?}"),
    );
    drop(x);
}

/// E4 — refresh by a raw host→device write into the buffer (`client.write`,
/// same pointer), reached through the `extension` primitive. Needs only `&`
/// access, so the closure may even hold a clone.
#[cfg(feature = "cubecl")]
fn e4_client_write(device: &Device) {
    use burn_cubecl::cubecl::bytes::Bytes;
    println!("E4 refresh via client.write (closure holds a clone, Int token)");
    let tok = Tensor::<1, Int>::from_ints([1, 2, 3], device);
    let dtype = tok.dtype();
    let held = tok.clone();
    let mut graph = capture(device, move || held.clone().float().mul_scalar(2.0));
    let id0 = buffer_id(&tok);
    for k in 1..=3i64 {
        let new = [7 * k, -k, 100 + k];
        let bytes = match dtype {
            burn::tensor::DType::I64 => Bytes::from_elems(new.to_vec()),
            burn::tensor::DType::I32 => Bytes::from_elems(new.iter().map(|&v| v as i32).collect()),
            other => panic!("unexpected int dtype {other:?}"),
        };
        let Ok(cube) = tok.clone().try_into_primitive::<burn_cubecl::Cube>() else {
            verdict("client.write", true, "not a cube tensor — skipped");
            return;
        };
        cube.client.write(&cube.handle, bytes);
        drop(cube);
        // Safety: `tok`'s buffer is owned by the closure inside `graph` too.
        let out = vals(unsafe { graph.replay() });
        let expected: Vec<f32> = new.iter().map(|&v| 2.0 * v as f32).collect();
        verdict(
            &format!("write {k}"),
            close(&out, &expected) && buffer_id(&tok) == id0,
            format!("out {out:?}, dtype {dtype:?}"),
        );
    }
}

/// E5 — the decode pattern: a state advanced **inside** the graph by an in-place
/// write-back (`s ← ½s + t`), and a token refreshed between replays. `capture`
/// runs the closure 4× (3 warm-up + 1 recorded), so the state is restored
/// afterwards; then N replays must track the host reference exactly.
fn e5_self_advancing(device: &Device) {
    const N: usize = 200;
    println!("E5 self-advancing state (in-graph write-back) + token refresh, {N} replays");
    let s0 = [1.0f32, -2.0, 0.25, 4.0];
    let state = RefCell::new(Some(Tensor::<1>::from_floats(s0, device)));
    let tok = RefCell::new(Some(Tensor::<1>::from_floats([0.0; 4], device)));
    let step = || {
        let s = state.borrow_mut().take().unwrap();
        let t = tok.borrow().as_ref().unwrap().clone();
        let next = s.clone().mul_scalar(0.5).add(t);
        // `s` is uniquely owned again (its clone was consumed), so this writes
        // into the stable buffer — recorded into the graph as a copy kernel.
        *state.borrow_mut() = Some(s.slice_assign([0..4], next.clone()));
        next
    };
    let id0 = buffer_id(state.borrow().as_ref().unwrap());
    let mut graph = capture(device, step);
    let after_capture = vals(state.borrow().as_ref().unwrap());
    // Host reference of the capture-time runs with t = 0: the recorded run only
    // records on a hardware graph, so 3 warm-ups there, 4 runs on the fallback.
    let runs = if graph.is_hardware() { 3 } else { 4 };
    let mut reference = s0;
    for _ in 0..runs {
        reference = reference.map(|v| 0.5 * v);
    }
    verdict(
        &format!("capture advanced the state {runs}×"),
        close(&after_capture, &reference) && buffer_id(state.borrow().as_ref().unwrap()) == id0,
        format!("{after_capture:?}"),
    );
    // Restore the pre-capture state, in place.
    let refresh = |slot: &RefCell<Option<Tensor<1>>>, v: [f32; 4]| {
        let mut s = slot.borrow_mut();
        let x = s.take().unwrap();
        *s = Some(x.slice_assign([0..4], Tensor::from_floats(v, device)));
    };
    refresh(&state, s0);
    let mut reference = s0;
    let (mut worst, mut ok) = (0.0f32, true);
    for k in 0..N {
        let t = [k as f32 * 0.01, 1.0, -(k as f32).sin(), 0.5];
        refresh(&tok, t);
        // Safety: `state`/`tok` outlive `graph` and keep their buffers (every
        // write is in place); one thread, so all work is on the capture stream.
        let out = vals(unsafe { graph.replay() });
        for i in 0..4 {
            reference[i] = 0.5 * reference[i] + t[i];
        }
        worst = out.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(worst, f32::max);
        ok &= close(&out, &reference);
    }
    let fin = vals(state.borrow().as_ref().unwrap());
    verdict(&format!("{N} replays track the reference"), ok, format!("max |Δ| {worst:e}"));
    verdict(
        "stable state == last output, same buffer",
        close(&fin, &reference) && buffer_id(state.borrow().as_ref().unwrap()) == id0,
        format!("{fin:?}"),
    );
}

/// E6 — what a replay buys: a chain of `ops` tiny elementwise launches, eager vs
/// replay, each iteration synced (the decode regime: one step, then the host
/// needs the result).
fn e6_timing(device: &Device, ops: usize, iters: usize) {
    println!("E6 timing: chain of {ops} elementwise ops on [64], {iters} iterations");
    let x = Tensor::<1>::ones([64], device);
    let chain = || {
        let mut y = x.clone();
        for i in 0..ops {
            y = if i % 2 == 0 { y.mul_scalar(1.0001) } else { y.add_scalar(-0.0001) };
        }
        y
    };
    for _ in 0..5 {
        let _ = chain();
    }
    device.sync().unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let y = chain();
        device.sync().unwrap();
        drop(y);
    }
    let eager = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let mut graph = capture(device, chain);
    let t = Instant::now();
    for _ in 0..iters {
        // Safety: `x` outlives `graph`; one thread.
        unsafe { graph.replay() };
        device.sync().unwrap();
    }
    let replay = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "  eager {eager:.3} ms/iter ({:.1} µs/op) | replay {replay:.3} ms/iter | ×{:.1} (hardware: {})",
        eager * 1e3 / ops as f64,
        eager / replay,
        graph.is_hardware()
    );
}
