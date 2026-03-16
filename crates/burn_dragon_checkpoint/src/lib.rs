#![recursion_limit = "256"]

//! Shared checkpoint/deployment helpers for Dragon models.
//!
//! Preferred library-facing surface:
//! - [`api::burnpack`] for burnpack save/load/splitting helpers
//! - [`api::policy`] for precision-aware candidate selection
//! - [`api::expert`] only when lower-level manifest helpers are needed

pub mod bootstrap;
pub mod bundle;
pub mod parts;
pub mod policy;
pub mod precision;
pub mod run;

pub mod api {
    //! Curated checkpoint/deployment surface.

    pub mod bootstrap {
        pub use crate::bootstrap::{
            BurnpackBootstrapConfig, candidate_burnpack_urls, default_cache_root,
            resolve_or_bootstrap_burnpack, resolve_or_bootstrap_burnpack_with_progress,
        };
    }

    pub mod bundle {
        pub use crate::bundle::{
            BurnpackBundleExportOptions, BurnpackBundleExportReport,
            export_model_to_burnpack_bundle,
        };
    }

    pub mod burnpack {
        pub use crate::parts::{
            BurnpackPartEntry, BurnpackPartsManifest, BurnpackPartsReport,
            apply_burnpack_part_bytes, apply_burnpack_parts_bytes_with_progress,
            burnpack_parts_manifest_path, ensure_burnpack_parts,
            load_model_from_burnpack_candidates, load_model_from_burnpack_candidates_with_progress,
            load_model_from_burnpack_file, load_model_from_burnpack_part_bytes,
            load_model_from_burnpack_part_bytes_with_progress, load_model_from_burnpack_parts,
            load_model_from_burnpack_parts_with_progress, manifest_is_complete,
            read_parts_manifest, resolve_part_entry_path, save_model_to_burnpack,
            save_model_to_burnpack_with_parts, try_load_model_from_burnpack_candidates,
            try_load_model_from_burnpack_candidates_with_progress,
            try_load_model_from_burnpack_parts, try_load_model_from_burnpack_parts_with_progress,
            write_burnpack_parts,
        };
        pub use crate::precision::{
            BurnpackFloatPrecision, convert_burnpack_precision, dtype_precision_label,
        };
    }

    pub mod policy {
        pub use crate::policy::{
            BurnpackLoadPolicy, BurnpackPrecisionPreference, burnpack_path,
            candidate_burnpack_paths,
        };
    }

    pub mod run {
        pub use crate::run::{
            CheckpointExportReport, checkpoint_bin_path, format_checkpoint_load_error,
            load_json_snapshot, resolve_checkpoint_base, resolve_checkpoint_run_dir,
            run_snapshot_path, write_json_snapshot,
        };
    }

    pub mod expert {
        pub use crate::{bootstrap, bundle, parts, policy, precision, run};
    }
}

pub use bootstrap::{
    BurnpackBootstrapConfig, candidate_burnpack_urls, default_cache_root,
    resolve_or_bootstrap_burnpack, resolve_or_bootstrap_burnpack_with_progress,
};
pub use bundle::{
    BurnpackBundleExportOptions, BurnpackBundleExportReport, export_model_to_burnpack_bundle,
};
pub use parts::{
    BurnpackPartEntry, BurnpackPartsManifest, BurnpackPartsReport, apply_burnpack_part_bytes,
    apply_burnpack_parts_bytes_with_progress, burnpack_parts_manifest_path, ensure_burnpack_parts,
    load_model_from_burnpack_candidates, load_model_from_burnpack_candidates_with_progress,
    load_model_from_burnpack_file, load_model_from_burnpack_part_bytes,
    load_model_from_burnpack_part_bytes_with_progress, load_model_from_burnpack_parts,
    load_model_from_burnpack_parts_with_progress, manifest_is_complete, read_parts_manifest,
    resolve_part_entry_path, save_model_to_burnpack, save_model_to_burnpack_with_parts,
    try_load_model_from_burnpack_candidates, try_load_model_from_burnpack_candidates_with_progress,
    try_load_model_from_burnpack_parts, try_load_model_from_burnpack_parts_with_progress,
    write_burnpack_parts,
};
pub use policy::{
    BurnpackLoadPolicy, BurnpackPrecisionPreference, burnpack_path, candidate_burnpack_paths,
};
pub use precision::{BurnpackFloatPrecision, convert_burnpack_precision, dtype_precision_label};
pub use run::{
    CheckpointExportReport, checkpoint_bin_path, format_checkpoint_load_error, load_json_snapshot,
    resolve_checkpoint_base, resolve_checkpoint_run_dir, run_snapshot_path, write_json_snapshot,
};
