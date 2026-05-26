//! Macros to cut down the per-backend boilerplate for `*SsdBackendExt` traits.
//!
//! Both Mamba-2 and Mamba-3 define an SSD-backend extension trait whose default
//! implementation already does the right thing for every burn backend except
//! `Autodiff` (which gets the custom backward). The macros below emit the
//! per-backend "use the default impl" blocks and the autodiff marker trait.

/// Emit `impl $trait_name for <backend> {}` blocks for every burn backend
/// supported by this crate, opting in to the trait's default body.
///
/// Each impl is feature-gated by the corresponding `backend-*` / `cubecl` /
/// `fusion` feature.
#[macro_export]
macro_rules! impl_ssd_backend_ext_for_burn_backends {
    ($trait_name:path) => {
        #[cfg(feature = "backend-ndarray")]
        impl<F, I> $trait_name for burn::backend::NdArray<F, I> {}

        #[cfg(feature = "backend-flex")]
        impl $trait_name for burn::backend::Flex {}

        #[cfg(any(feature = "backend-tch-cpu", feature = "backend-tch-gpu"))]
        impl<F, I> $trait_name for burn::backend::libtorch::LibTorch<F, I> {}

        #[cfg(feature = "backend-remote")]
        impl<F, I> $trait_name for burn::backend::RemoteBackend<F, I> {}

        #[cfg(feature = "cubecl")]
        impl<R, F, I, BT> $trait_name for burn_cubecl::CubeBackend<R, F, I, BT>
        where
            R: burn_cubecl::CubeRuntime,
            F: burn_cubecl::FloatElement,
            I: burn_cubecl::IntElement,
            BT: burn_cubecl::element::BoolElement,
        {
        }

        // Fusion delegates to the inner backend's default impl.
        #[cfg(feature = "fusion")]
        impl<B: burn_fusion::FusionBackend + $trait_name> $trait_name for burn_fusion::Fusion<B> {}
    };
}

/// Declare a marker trait `$autodiff_trait: Backend + $ext_trait + AutodiffBackend`
/// and a blanket impl for `burn::backend::Autodiff<B>` whenever `B: $ext_trait`.
///
/// Both Mamba-2 and Mamba-3 expose this marker to let callers bound their own
/// generics on "autodiff-capable backend that also implements the SSD ext."
#[macro_export]
macro_rules! decl_ssd_autodiff_backend_ext {
    ($autodiff_trait:ident, $ext_trait:path $(, $extra_bound:path)*) => {
        #[cfg(feature = "autodiff")]
        pub trait $autodiff_trait:
            burn::tensor::backend::Backend
            + burn::tensor::backend::AutodiffBackend
            + $ext_trait
            $(+ $extra_bound)*
        {
        }
        /// Any autodiff-compatible backend whose inner backend implements the
        /// ext trait satisfies the marker. The actual custom backward lives in
        /// `super::backward` (a custom impl of the ext trait for `Autodiff<B>`).
        #[cfg(feature = "autodiff")]
        impl<B> $autodiff_trait
            for burn::backend::Autodiff<B>
            where
                B: burn::tensor::backend::Backend + $ext_trait $(+ $extra_bound)*
        {
        }
    };
}
