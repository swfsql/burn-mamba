//! TEMPORARY (graph-capture handoff, phase 2): capture one decode `step` of a
//! `MambaVocabNet` and replay it for a whole generation.
//!
//! usage: `graph-decode <model_config.json> [batch=1] [tokens=300] [warmup=20]`
//!
//! Random weights (parity and timing need none). Two runs over the same
//! teacher-forced random tokens, each opened by `prime` (the class latents):
//! - **eager** — `step` per token, logits read back to the host;
//! - **replay** — the first token eager, then one captured `step` whose closure
//!   writes its new cache back into the stable one
//!   ([`CacheTensors::assign_in_place`]); per token: refresh the token buffer,
//!   replay, read the logits.
//!
//! The replay's logits must match the eager ones; on cubecl the stable cache's
//! buffer ids must not move. Timing excludes the first `warmup` tokens.

use std::cell::RefCell;
use std::time::Instant;

use burn::config::Config;
use burn::prelude::*;
use burn::tensor::capture;
use burn_mamba::prelude::*;
use burn_stack::modules::TensorZip;
use burn_stack::utils::ClassCursors;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct Args {
    config: String,
    batch: usize,
    tokens: usize,
    warmup: usize,
}

fn args() -> Args {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, default: usize| a.get(i).map_or(default, |s| s.parse().unwrap());
    Args {
        config: a.first().expect("usage: graph-decode <model_config.json> [batch] [tokens] [warmup]").clone(),
        batch: num(1, 1),
        tokens: num(2, 300),
        warmup: num(3, 20),
    }
}

/// Per-token timings, split at the host: `enqueue` is the `step`/`replay` call
/// alone, `total` adds the token upload and the logits read (a sync).
#[derive(Default)]
struct Timing {
    enqueue_ms: f64,
    total_ms: f64,
    n: usize,
}

impl Timing {
    fn add(&mut self, enqueue: f64, total: f64) {
        self.enqueue_ms += enqueue;
        self.total_ms += total;
        self.n += 1;
    }
    fn report(&self, label: &str) -> f64 {
        let (e, t) = (self.enqueue_ms / self.n as f64, self.total_ms / self.n as f64);
        println!("  {label:7} enqueue {e:7.3} ms/tok | total {t:7.3} ms/tok | {:8.1} tok/s", 1e3 / t);
        t
    }
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn read(t: &Tensor<2>) -> Vec<f32> {
    t.clone().into_data().convert::<f32>().try_to_vec::<f32>().unwrap()
}

/// Write `ids` into `slot`'s buffer: `client.write` on cubecl (host→device, same
/// pointer), an in-place `slice_assign` elsewhere.
fn refresh_tokens(slot: &RefCell<Option<Tensor<1, Int>>>, ids: &[i32]) {
    #[cfg(feature = "cubecl")]
    {
        use burn_cubecl::cubecl::bytes::Bytes;
        let t = slot.borrow().as_ref().unwrap().clone();
        let dtype = t.dtype();
        if let Ok(cube) = t.try_into_primitive::<burn_cubecl::Cube>() {
            let bytes = match dtype {
                burn::tensor::DType::I32 => Bytes::from_elems(ids.to_vec()),
                burn::tensor::DType::I64 => Bytes::from_elems(ids.iter().map(|&v| v as i64).collect()),
                other => panic!("unexpected int dtype {other:?}"),
            };
            cube.client.write(&cube.handle, bytes);
            return;
        }
    }
    let mut s = slot.borrow_mut();
    let t = s.take().unwrap();
    let device = t.device();
    *s = Some(t.slice_assign([0..ids.len()], Tensor::from_ints(ids, &device)));
}

/// The buffer ids of every tensor in `caches` (cubecl only; empty elsewhere).
fn buffer_ids(caches: &MambaCaches) -> Vec<usize> {
    struct Ids(Vec<usize>);
    impl TensorZip for Ids {
        fn zip<const D: usize>(&mut self, a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
            drop(b);
            #[cfg(feature = "cubecl")]
            if let Ok(cube) = a.clone().try_into_primitive::<burn_cubecl::Cube>() {
                self.0.push(cube.handle.memory.descriptor().id.value);
            }
            a
        }
    }
    let mut ids = Ids(Vec::new());
    let _ = caches.clone().zip_tensors(caches.clone(), &mut ids);
    ids.0
}

fn eager(net: &MambaVocabNet, tokens: &[Vec<i32>], warmup: usize, device: &Device) -> (Vec<Vec<f32>>, Timing) {
    let batch = tokens[0].len();
    let mut class = ClassCursors::stream();
    let (_, mut caches) = net.prime(batch, None, Some(&mut class));
    let (mut logits, mut timing) = (Vec::new(), Timing::default());
    for (k, ids) in tokens.iter().enumerate() {
        let t0 = Instant::now();
        let x = Tensor::<1, Int>::from_ints(ids.as_slice(), device);
        let t1 = Instant::now();
        let (y, c) = net.step(x, caches.take(), Some(&mut class));
        let enqueue = ms(t1);
        logits.push(read(&y));
        if k >= warmup {
            timing.add(enqueue, ms(t0));
        }
        caches = Some(c);
    }
    (logits, timing)
}

fn replay(net: &MambaVocabNet, tokens: &[Vec<i32>], warmup: usize, device: &Device) -> (Vec<Vec<f32>>, Timing) {
    let batch = tokens[0].len();
    let mut class = ClassCursors::stream();
    let (_, caches) = net.prime(batch, None, Some(&mut class));
    // The first token eager: `step` from a warm cache from here on, and the
    // class latents are all spent, so the captured step's plan is empty.
    let x = Tensor::<1, Int>::from_ints(tokens[0].as_slice(), device);
    let (y, caches) = net.step(x, caches, Some(&mut class));
    let mut logits = vec![read(&y)];

    let stable = RefCell::new(Some(caches.into_owned_buffers()));
    let token = RefCell::new(Some(Tensor::<1, Int>::from_ints(tokens[1].as_slice(), device)));
    let ids0 = buffer_ids(stable.borrow().as_ref().unwrap());
    let snapshot = stable.borrow().as_ref().unwrap().clone().into_owned_buffers();

    let t = Instant::now();
    let mut graph = capture(device, || {
        let caches = stable.borrow_mut().take().unwrap();
        let x = token.borrow().as_ref().unwrap().clone();
        let (y, new) = net.step(x, Some(caches.clone()), None);
        *stable.borrow_mut() = Some(caches.assign_in_place(new));
        y
    });
    println!("  capture {:.1} ms, hardware: {}", ms(t), graph.is_hardware());
    // The capture ran the closure (3 warm-ups executed on a hardware graph, 4
    // on the fallback): put the pre-capture state back, in place.
    {
        let mut s = stable.borrow_mut();
        let c = s.take().unwrap();
        *s = Some(c.assign_in_place(snapshot));
    }
    let ids1 = buffer_ids(stable.borrow().as_ref().unwrap());

    let mut timing = Timing::default();
    for (k, ids) in tokens.iter().enumerate().skip(1) {
        let t0 = Instant::now();
        refresh_tokens(&token, ids);
        let t1 = Instant::now();
        // Safety: every buffer the graph touched is held by `stable`, `token`,
        // `net` or the graph itself, all outliving it, and every refresh is in
        // place (checked by the buffer ids); one thread, so one stream.
        let y = unsafe { graph.replay() }.clone();
        let enqueue = ms(t1);
        logits.push(read(&y));
        if k >= warmup {
            timing.add(enqueue, ms(t0));
        }
    }
    let ids2 = buffer_ids(stable.borrow().as_ref().unwrap());
    println!(
        "  stable cache: {} tensors, buffers kept across capture: {}, across replays: {}",
        ids0.len(),
        ids0 == ids1,
        ids1 == ids2
    );
    (logits, timing)
}

fn main() {
    let args = args();
    let device = Device::default();
    let config = MambaVocabNetConfig::load(&args.config).expect("model config");
    let net = config.init(&device);
    println!("device: {device:?} | batch {} | tokens {} | warmup {}", args.batch, args.tokens, args.warmup);

    let x = Tensor::<1, Int>::zeros([args.batch], &device);
    let vocab = net.step(x, None, None).0.dims()[1];
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let tokens: Vec<Vec<i32>> = (0..args.tokens)
        .map(|_| (0..args.batch).map(|_| rng.random_range(0..vocab as i32)).collect())
        .collect();

    println!("eager:");
    let (ref_logits, eager_t) = eager(&net, &tokens, args.warmup, &device);
    println!("replay:");
    let (got_logits, replay_t) = replay(&net, &tokens, args.warmup, &device);

    let mut worst = 0.0f32;
    let mut worst_at = 0;
    for (k, (a, b)) in ref_logits.iter().zip(&got_logits).enumerate() {
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        if d > worst {
            (worst, worst_at) = (d, k);
        }
    }
    println!("parity: max |Δlogits| {worst:e} (at token {worst_at}) over {} tokens", ref_logits.len());
    let e = eager_t.report("eager");
    let r = replay_t.report("replay");
    println!("  speed-up ×{:.2}", e / r);
}
