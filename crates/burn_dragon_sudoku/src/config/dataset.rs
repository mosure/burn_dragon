use super::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuDatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
    #[serde(default)]
    pub augment: bool,
    #[serde(default = "default_augment_prob")]
    pub augment_prob: f32,
    #[serde(flatten)]
    pub source: SudokuDatasetSourceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SudokuDatasetSourceConfig {
    HuggingFace(SudokuHuggingFaceConfig),
    Local(SudokuLocalConfig),
}

impl Default for SudokuDatasetSourceConfig {
    fn default() -> Self {
        Self::HuggingFace(SudokuHuggingFaceConfig::default())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuHuggingFaceConfig {
    pub repo_id: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub format: SudokuRecordFormat,
    #[serde(default = "default_hf_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_puzzle_field")]
    pub puzzle_field: String,
    #[serde(default = "default_solution_field")]
    pub solution_field: String,
    #[serde(default)]
    pub train_max_records: Option<usize>,
    #[serde(default)]
    pub validation_max_records: Option<usize>,
    #[serde(default)]
    pub max_records: Option<usize>,
}

impl Default for SudokuHuggingFaceConfig {
    fn default() -> Self {
        Self {
            repo_id: "Ritvik19/Sudoku-Dataset".to_string(),
            token: None,
            revision: None,
            format: SudokuRecordFormat::Parquet,
            train_files: default_hf_train_files(),
            validation_files: vec!["valid_0.parquet".to_string()],
            puzzle_field: default_puzzle_field(),
            solution_field: default_solution_field(),
            train_max_records: None,
            validation_max_records: None,
            max_records: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuLocalConfig {
    pub root: PathBuf,
    #[serde(default)]
    pub format: SudokuRecordFormat,
    #[serde(default = "default_local_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_puzzle_field")]
    pub puzzle_field: String,
    #[serde(default = "default_solution_field")]
    pub solution_field: String,
    #[serde(default)]
    pub train_max_records: Option<usize>,
    #[serde(default)]
    pub validation_max_records: Option<usize>,
    #[serde(default)]
    pub max_records: Option<usize>,
}

impl Default for SudokuLocalConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("data/sudoku"),
            format: SudokuRecordFormat::Jsonl,
            train_files: default_local_train_files(),
            validation_files: Vec::new(),
            puzzle_field: default_puzzle_field(),
            solution_field: default_solution_field(),
            train_max_records: None,
            validation_max_records: None,
            max_records: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuRecordFormat {
    #[default]
    Jsonl,
    Csv,
    Parquet,
}
