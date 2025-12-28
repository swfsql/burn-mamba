use crate::common::backend::Element;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn_dataset::network::downloader::download_file_as_bytes;
use burn_dataset::{
    Dataset, InMemDataset,
    transform::{Mapper, MapperDataset},
};
use flate2::read::GzDecoder;
use num_traits::AsPrimitive;
use serde::{Deserialize, Serialize};
use std::fs::{File, create_dir_all};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

// from the vision source
// https://github.com/tracel-ai/burn/blob/fa4f9845a6b2279cd8de68bf7ca5a7eb76dec96d/crates/burn-dataset/src/vision/mnist.rs
//
// note: activating the "vision" feature for burn-dataset would considerably increase the dependencies,
// so the necessary code was replicated here

// CVDF mirror of http://yann.lecun.com/exdb/mnist/
const URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

pub const WIDTH: usize = 28;
pub const HEIGHT: usize = 28;

/// MNIST item.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct MnistFlatItem {
    /// Image as a flat array of floats.
    /// Each value is a brightness, in between 0.0 and 255.0.
    ///
    /// # Shape
    /// [WIDTH * HEIGHT]
    pub image: Vec<Element>,

    /// Label of the image.  
    /// Each value is in between 0 and 9.
    pub label: u8,
}

#[derive(Deserialize, Debug, Clone)]
struct MnistFlatItemRaw {
    pub image_bytes: Vec<u8>,
    pub label: u8,
}

struct BytesToFlatImage;

impl Mapper<MnistFlatItemRaw, MnistFlatItem> for BytesToFlatImage {
    /// Convert a raw MNIST item (image bytes) to a MNIST item (flat array image).
    fn map(&self, item: &MnistFlatItemRaw) -> MnistFlatItem {
        // Ensure the image dimensions are correct.
        debug_assert_eq!(item.image_bytes.len(), WIDTH * HEIGHT);

        // Convert the image to a flat array of floats.
        let image: Vec<Element> = item
            .image_bytes
            .iter()
            .map(|brightness| {
                let element: Element = (*brightness).as_();
                element
            })
            .collect();

        MnistFlatItem {
            image,
            label: item.label,
        }
    }
}

type MappedDataset =
    MapperDataset<InMemDataset<MnistFlatItemRaw>, BytesToFlatImage, MnistFlatItemRaw>;

/// The MNIST dataset consists of 70,000 28x28 black-and-white images in 10 classes (one for each digits), with 7,000
/// images per class. There are 60,000 training images and 10,000 test images.
///
/// The data is downloaded from the web from the [CVDF mirror](https://github.com/cvdfoundation/mnist).
pub struct MnistDataset {
    dataset: MappedDataset,
}

impl Dataset<MnistFlatItem> for MnistDataset {
    fn get(&self, index: usize) -> Option<MnistFlatItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MnistDataset {
    /// Creates a new train dataset.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Creates a new test dataset.
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        // Download dataset
        let root = MnistDataset::download(split);

        // MNIST is tiny so we can load it in-memory
        // Train images (u8): 28 * 28 * 60000 = 47.04Mb
        // Test images (u8): 28 * 28 * 10000 = 7.84Mb
        let images = MnistDataset::read_images(&root, split);
        let labels = MnistDataset::read_labels(&root, split);

        // Collect as vector of MnistItemRaw
        let items: Vec<_> = images
            .into_iter()
            .zip(labels)
            .map(|(image_bytes, label)| MnistFlatItemRaw { image_bytes, label })
            .collect();

        let dataset = InMemDataset::new(items);
        let dataset = MapperDataset::new(dataset, BytesToFlatImage);

        Self { dataset }
    }

    /// Download the MNIST dataset files from the web.
    /// Panics if the download cannot be completed or the content of the file cannot be written to disk.
    fn download(split: &str) -> PathBuf {
        // Dataset files are stored un the burn-dataset cache directory
        let cache_dir = dirs::home_dir()
            .expect("Could not get home directory")
            .join(".cache")
            .join("burn-dataset");
        let split_dir = cache_dir.join("mnist").join(split);

        if !split_dir.exists() {
            create_dir_all(&split_dir).expect("Failed to create base directory");
        }

        // Download split files
        match split {
            "train" => {
                MnistDataset::download_file(TRAIN_IMAGES, &split_dir);
                MnistDataset::download_file(TRAIN_LABELS, &split_dir);
            }
            "test" => {
                MnistDataset::download_file(TEST_IMAGES, &split_dir);
                MnistDataset::download_file(TEST_LABELS, &split_dir);
            }
            _ => panic!("Invalid split specified {split}"),
        };

        split_dir
    }

    /// Download a file from the MNIST dataset URL to the destination directory.
    /// File download progress is reported with the help of a [progress bar](indicatif).
    fn download_file<P: AsRef<Path>>(name: &str, dest_dir: &P) -> PathBuf {
        // Output file name
        let file_name = dest_dir.as_ref().join(name);

        if !file_name.exists() {
            // Download gzip file
            let bytes = download_file_as_bytes(&format!("{URL}{name}.gz"), name);

            // Create file to write the downloaded content to
            let mut output_file = File::create(&file_name).unwrap();

            // Decode gzip file content and write to disk
            let mut gz_buffer = GzDecoder::new(&bytes[..]);
            std::io::copy(&mut gz_buffer, &mut output_file).unwrap();
        }

        file_name
    }

    /// Read images at the provided path for the specified split.
    /// Each image is a vector of bytes.
    fn read_images<P: AsRef<Path>>(root: &P, split: &str) -> Vec<Vec<u8>> {
        let file_name = if split == "train" {
            TRAIN_IMAGES
        } else {
            TEST_IMAGES
        };
        let file_name = root.as_ref().join(file_name);

        // Read number of images from 16-byte header metadata
        let mut f = File::open(file_name).unwrap();
        let mut buf = [0u8; 4];
        let _ = f.seek(SeekFrom::Start(4)).unwrap();
        f.read_exact(&mut buf)
            .expect("Should be able to read image file header");
        let size = u32::from_be_bytes(buf);

        let mut buf_images: Vec<u8> = vec![0u8; WIDTH * HEIGHT * (size as usize)];
        let _ = f.seek(SeekFrom::Start(16)).unwrap();
        f.read_exact(&mut buf_images)
            .expect("Should be able to read image file header");

        buf_images
            .chunks(WIDTH * HEIGHT)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    /// Read labels at the provided path for the specified split.
    fn read_labels<P: AsRef<Path>>(root: &P, split: &str) -> Vec<u8> {
        let file_name = if split == "train" {
            TRAIN_LABELS
        } else {
            TEST_LABELS
        };
        let file_name = root.as_ref().join(file_name);

        // Read number of labels from 8-byte header metadata
        let mut f = File::open(file_name).unwrap();
        let mut buf = [0u8; 4];
        let _ = f.seek(SeekFrom::Start(4)).unwrap();
        f.read_exact(&mut buf)
            .expect("Should be able to read label file header");
        let size = u32::from_be_bytes(buf);

        let mut buf_labels: Vec<u8> = vec![0u8; size as usize];
        let _ = f.seek(SeekFrom::Start(8)).unwrap();
        f.read_exact(&mut buf_labels)
            .expect("Should be able to read labels from file");

        buf_labels
    }
}

// from the rust book
// https://burn.dev/books/burn/basic-workflow/data.html#data

#[derive(Clone, Default)]
pub struct MnistBatcher {}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    /// The input feature is the brightness, z-score normalized (mean=0.0, stddev=1.0).  
    /// The original dataset had mean=0.1307, stddev=0.3081.  
    ///
    /// The mappings are:
    ///
    /// * `z = (value / 255 - mean) / stddev`,  
    /// * `value = (z * stddev + mean) * 255`.
    ///
    /// # Shape
    /// [batch_size, WIDTH * HEIGHT, 1]
    pub images: Tensor<B, 3>,
    /// # Shape
    /// [batch_size]
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<B, MnistFlatItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistFlatItem>, device: &B::Device) -> MnistBatch<B> {
        let (items_image, items_label): (Vec<_>, Vec<_>) = items
            .into_iter()
            .map(|item| (item.image, item.label))
            .unzip();
        let images = items_image
            .into_iter()
            .map(|image: Vec<Element>| {
                TensorData::new(image, [1, WIDTH * HEIGHT]).convert::<B::FloatElem>()
            })
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor: Tensor<B, 2>| tensor.reshape([1, WIDTH * HEIGHT, 1]))
            // Normalize: scale between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items_label
            .into_iter()
            .map(|label: u8| {
                Tensor::<B, 1, Int>::from_data([(label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
