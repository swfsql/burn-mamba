//! Graph-capture support for the caches: the
//! [`CacheTensors`](burn_stack::modules::CacheTensors) traversal of each
//! family. With it, a captured `step`
//! ([`CapturedStep`](burn_stack::utils::graph::CapturedStep)) writes its new
//! cache **into** the buffers of the old one.
//!
//! There is one traversal per cache type, and it visits every tensor field. An
//! optional field must be present in both caches or in neither. This module
//! also shows why the stable cache must first be copied into buffers of its
//! own ([`into_owned_buffers`](burn_stack::modules::CacheTensors::into_owned_buffers)):
//! the Mamba-3 tap slots come back from `step` as `narrow`s of its own
//! tensors.

use burn::prelude::*;
use burn_stack::modules::{CacheTensors, TensorZip};

impl CacheTensors for super::MambaCaches {
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
        match (self, other) {
            #[cfg(feature = "mamba1")]
            (Self::Mamba1(a), Self::Mamba1(b)) => Self::Mamba1(a.zip_tensors(b, z)),
            #[cfg(feature = "mamba2")]
            (Self::Mamba2(a), Self::Mamba2(b)) => Self::Mamba2(a.zip_tensors(b, z)),
            #[cfg(feature = "mamba3")]
            (Self::Mamba3(a), Self::Mamba3(b)) => Self::Mamba3(a.zip_tensors(b, z)),
            #[allow(unreachable_patterns)]
            _ => panic!("the two caches differ in family"),
        }
    }
}

#[cfg(feature = "mamba1")]
mod impl_mamba1 {
    use super::*;
    use crate::mamba1::prelude::{Mamba1Cache, Mamba1Caches};

    impl CacheTensors for Mamba1Cache {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                conv_bik: z.zip(self.conv_bik, other.conv_bik),
                ssm_bir: z.zip(self.ssm_bir, other.ssm_bir),
            }
        }
    }

    impl CacheTensors for Mamba1Caches {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                caches: self.caches.zip_tensors(other.caches, z),
            }
        }
    }
}

#[cfg(feature = "mamba2")]
mod impl_mamba2 {
    use super::*;
    use crate::mamba2::prelude::{Mamba2Cache, Mamba2Caches};

    impl CacheTensors for Mamba2Cache {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                conv_bvk: z.zip(self.conv_bvk, other.conv_bvk),
                ssm_bhpr: z.zip(self.ssm_bhpr, other.ssm_bhpr),
            }
        }
    }

    impl CacheTensors for Mamba2Caches {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                caches: self.caches.zip_tensors(other.caches, z),
            }
        }
    }
}

#[cfg(feature = "mamba3")]
mod impl_mamba3 {
    use super::*;
    use crate::mamba3::double_ssd::prelude::{Mamba3DoubleSsdCache, Mamba3DoubleSsdCaches};
    use crate::mamba3::prelude::{Mamba3Cache, Mamba3Caches, RotationState};
    use crate::mamba3::single_ssd::prelude::{Mamba3SingleSsdCache, Mamba3SingleSsdCaches};

    /// An optional field: both present or both absent.
    fn zip_option<const D: usize>(
        a: Option<Tensor<D>>,
        b: Option<Tensor<D>>,
        z: &mut impl TensorZip,
    ) -> Option<Tensor<D>> {
        match (a, b) {
            (Some(a), Some(b)) => Some(z.zip(a, b)),
            (None, None) => None,
            _ => panic!("the two caches differ in which optional fields they hold"),
        }
    }

    fn zip_rotation(a: RotationState, b: RotationState, z: &mut impl TensorZip) -> RotationState {
        match (a, b) {
            (RotationState::Real(u), RotationState::Real(_)) => RotationState::Real(u),
            (RotationState::Angle(a), RotationState::Angle(b)) => RotationState::Angle(z.zip(a, b)),
            (RotationState::Quaternion(a), RotationState::Quaternion(b)) => {
                RotationState::Quaternion(z.zip(a, b))
            }
            (RotationState::Rotor(a), RotationState::Rotor(b)) => RotationState::Rotor(z.zip(a, b)),
            _ => panic!("the two caches differ in rotation kind"),
        }
    }

    impl CacheTensors for Mamba3DoubleSsdCache {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                ssm_bhpr: z.zip(self.ssm_bhpr, other.ssm_bhpr),
                k_state_bumhr: zip_option(self.k_state_bumhr, other.k_state_bumhr, z),
                v_state_buhp: zip_option(self.v_state_buhp, other.v_state_buhp, z),
                rotation: zip_rotation(self.rotation, other.rotation, z),
                log_precision_bh: zip_option(self.log_precision_bh, other.log_precision_bh, z),
                tropical_bh: zip_option(self.tropical_bh, other.tropical_bh, z),
            }
        }
    }

    /// The same fields as the double-ssd cache (a lossless move, see
    /// `mamba3/cache.rs`), so it rides that traversal.
    impl CacheTensors for Mamba3SingleSsdCache {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Mamba3DoubleSsdCache::from(self)
                .zip_tensors(other.into(), z)
                .into()
        }
    }

    impl CacheTensors for Mamba3Cache {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            match (self, other) {
                (Self::DoubleSsd(a), Self::DoubleSsd(b)) => Self::DoubleSsd(a.zip_tensors(b, z)),
                (Self::SingleSsd(a), Self::SingleSsd(b)) => Self::SingleSsd(a.zip_tensors(b, z)),
                _ => panic!("the two caches differ in SSD pathway"),
            }
        }
    }

    impl CacheTensors for Mamba3DoubleSsdCaches {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                caches: self.caches.zip_tensors(other.caches, z),
            }
        }
    }

    impl CacheTensors for Mamba3SingleSsdCaches {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            Self {
                caches: self.caches.zip_tensors(other.caches, z),
            }
        }
    }

    impl CacheTensors for Mamba3Caches {
        fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self {
            match (self, other) {
                (Self::DoubleSsd(a), Self::DoubleSsd(b)) => Self::DoubleSsd(a.zip_tensors(b, z)),
                (Self::SingleSsd(a), Self::SingleSsd(b)) => Self::SingleSsd(a.zip_tensors(b, z)),
                _ => panic!("the two caches differ in SSD pathway"),
            }
        }
    }
}
