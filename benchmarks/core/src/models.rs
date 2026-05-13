use crate::augmentation::AugmentationInput;

#[derive(Debug, Clone)]
pub(crate) struct PostprocessJob {
    pub(crate) job_id: u64,
    pub(crate) payload: String,
}

#[derive(Debug, Clone)]
pub(crate) struct AugmentationJob {
    pub(crate) job_id: u64,
    pub(crate) input: AugmentationInput,
}
