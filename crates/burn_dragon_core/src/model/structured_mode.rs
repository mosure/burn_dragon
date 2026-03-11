#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuredStepMode {
    Observe,
    Refine,
    Predict,
}

impl StructuredStepMode {
    pub const COUNT: usize = 3;

    pub const fn index(self) -> usize {
        match self {
            Self::Observe => 0,
            Self::Refine => 1,
            Self::Predict => 2,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Refine => "refine",
            Self::Predict => "predict",
        }
    }

    pub const fn temporal_dt(self) -> usize {
        match self {
            Self::Observe => 0,
            Self::Refine => 0,
            Self::Predict => 1,
        }
    }

    pub const fn resets_prediction_age(self) -> bool {
        matches!(self, Self::Observe)
    }
}
