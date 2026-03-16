#![recursion_limit = "256"]

//! Payload-agnostic stream, reset, and TBPTT contracts for Dragon models.
//!
//! Preferred library-facing surface:
//! - [`api::ids`]
//! - [`api::boundary`]
//! - [`api::window`]
//! - [`api::segment`]
//! - [`api::dataset`]
//! - [`api::cursor`]
//! - [`api::collate`]

pub mod alignment;
pub mod boundary;
pub mod collate;
pub mod cursor;
pub mod dataset;
pub mod ids;
pub mod policy;
pub mod segment;
pub mod window;

pub mod api {
    //! Curated stream/data surface.

    pub mod ids {
        pub use crate::ids::StreamSampleId;
    }

    pub mod boundary {
        pub use crate::boundary::StreamBoundary;
    }

    pub mod alignment {
        pub use crate::alignment::{
            StreamWindowSelection, TargetAlignmentSelection, resolve_stream_target_alignment,
            resolve_stream_window_alignment, resolve_target_alignment,
        };
    }

    pub mod window {
        pub use crate::window::TbpttWindow;
    }

    pub mod segment {
        pub use crate::segment::{CollatedStreamBatch, StreamSegment, StreamStepMetadata};
    }

    pub mod policy {
        pub use crate::policy::{FusionCarryPolicy, StateCarryPolicy, TargetAlignmentPolicy};
    }

    pub mod dataset {
        pub use crate::dataset::{InMemoryStreamDataset, StreamDataset};
    }

    pub mod cursor {
        pub use crate::cursor::{StreamCursor, VecStreamCursor};
    }

    pub mod collate {
        pub use crate::collate::{StreamCollatable, collate_stream_segments};
    }

    pub mod expert {
        pub use crate::{
            alignment, boundary, collate, cursor, dataset, ids, policy, segment, window,
        };
    }
}

pub use alignment::{
    StreamWindowSelection, TargetAlignmentSelection, resolve_stream_target_alignment,
    resolve_stream_window_alignment, resolve_target_alignment,
};
pub use boundary::StreamBoundary;
pub use collate::{StreamCollatable, collate_stream_segments};
pub use cursor::{StreamCursor, VecStreamCursor};
pub use dataset::{InMemoryStreamDataset, StreamDataset};
pub use ids::StreamSampleId;
pub use policy::{FusionCarryPolicy, StateCarryPolicy, TargetAlignmentPolicy};
pub use segment::{CollatedStreamBatch, StreamSegment, StreamStepMetadata};
pub use window::TbpttWindow;
