use crate::common::mnist::dataset::{HEIGHT, MnistBatch, WIDTH};
use crate::common::model::{MyMamba3Network, MyMamba3NetworkConfig};
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn_dataset::Dataset;
use burn_mamba::prelude::Mamba3BackendExt;
use burn_mamba::prelude::Mamba3SsdPath;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;
use std::sync::Mutex;

pub const NUM_CLASSES: usize = 10;

// ---------------------------------------------------------------------------
// Item — pure presence token
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GaussianItem;

// ---------------------------------------------------------------------------
// Dataset — only tracks length
// ---------------------------------------------------------------------------

pub struct GaussianDataset {
    len: usize,
}

impl GaussianDataset {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
    pub fn train(len: usize) -> Self {
        Self::new(len)
    }
    pub fn test(len: usize) -> Self {
        Self::new(len)
    }
}

impl Dataset<GaussianItem> for GaussianDataset {
    fn get(&self, index: usize) -> Option<GaussianItem> {
        (index < self.len).then_some(GaussianItem)
    }
    fn len(&self) -> usize {
        self.len
    }
}

// ---------------------------------------------------------------------------
// Batch + Batcher
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GaussianBatch<B: Backend> {
    pub inputs: Tensor<B, 4>, // [batch_size, HEIGHT, WIDTH, 1]
    pub logits: Tensor<B, 4>, // [batch_size, HEIGHT, WIDTH, NUM_CLASSES]
}

impl<B: Backend> GaussianBatch<B> {
    pub fn to_device<B2: Backend>(&self, device: &B2::Device) -> GaussianBatch<B2> {
        GaussianBatch {
            inputs: Tensor::<B2, 4>::from_data(self.inputs.to_data(), device),
            logits: Tensor::<B2, 4>::from_data(self.logits.to_data(), device),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GaussianBatcher<B: Backend> {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    teacher: MyMamba3Network<B>,
}

impl<B: Backend> GaussianBatcher<B> {
    pub fn new(seed: u64, teacher: MyMamba3Network<B>) -> Self {
        Self {
            rng: Arc::new(Mutex::new(rand::rngs::StdRng::seed_from_u64(seed))),
            teacher,
        }
    }
}

impl<B: Backend + Mamba3BackendExt> Batcher<B, GaussianItem, GaussianBatch<B>>
    for GaussianBatcher<B>
{
    fn batch(&self, items: Vec<GaussianItem>, device: &B::Device) -> GaussianBatch<B> {
        let batch_size = items.len();
        let normal = Normal::new(0., 1.).expect("Valid normal distribution");
        let mut rng = self.rng.lock().expect("rng lock poisoned");
        let sequence_size = HEIGHT * WIDTH;

        let inputs: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|_| {
                let pixels: Vec<f32> = (0..sequence_size)
                    .map(|_| f32::clamp(normal.sample(&mut *rng), 0., 1.))
                    .collect();
                let data = TensorData::new(pixels, [1, HEIGHT, WIDTH, 1]).convert::<B::FloatElem>();
                Tensor::<B, 4>::from_data(data, device)
            })
            .collect();

        let inputs = Tensor::cat(inputs, 0);
        let inputs_ = inputs.clone().reshape([batch_size, sequence_size, 1]);

        let ssd_path = Mamba3SsdPath::Minimal(None);
        let (output, _caches) = self.teacher.forward(inputs_, None, ssd_path);
        assert_eq!([batch_size, sequence_size, 10], output.dims());

        let output = output.reshape([batch_size, HEIGHT, WIDTH, 10]);

        GaussianBatch {
            inputs,
            logits: output,
        }
    }
}
