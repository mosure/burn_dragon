use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use csv::ReaderBuilder;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rand::prelude::*;
use serde_json::Value;

use crate::config::{
    SudokuDatasetConfig, SudokuDatasetSourceConfig, SudokuHuggingFaceConfig, SudokuLocalConfig,
    SudokuRecordFormat,
};
use crate::vocab::{SudokuVocab, GRID_LEN};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SudokuSplit {
    Train,
    Val,
}

#[derive(Clone)]
pub struct SudokuBatch<B: Backend> {
    pub puzzles: Tensor<B, 2, Int>,
    pub solutions: Tensor<B, 2, Int>,
}

impl<B: Backend> SudokuBatch<B> {
    pub fn new(puzzles: Tensor<B, 2, Int>, solutions: Tensor<B, 2, Int>) -> Self {
        Self { puzzles, solutions }
    }
}

#[derive(Clone, Debug)]
struct SudokuRecord {
    puzzle: Vec<u8>,
    solution: Vec<u8>,
}

#[derive(Clone)]
pub struct SudokuDataset {
    puzzles: Vec<Vec<u8>>,
    solutions: Vec<Vec<u8>>,
    train_len: usize,
    batch_size: usize,
    train_split_ratio: f32,
}

impl SudokuDataset {
    pub fn new(
        config: &SudokuDatasetConfig,
        batch_size: usize,
    ) -> Result<(Self, String), anyhow::Error> {
        let cache_dir = &config.cache_dir;
        fs::create_dir_all(cache_dir)?;

        let (train_records, mut val_records) = match &config.source {
            SudokuDatasetSourceConfig::HuggingFace(cfg) => {
                load_hf_records(cfg, cache_dir)?
            }
            SudokuDatasetSourceConfig::Local(cfg) => {
                load_local_records(cfg)?
            }
        };

        if train_records.is_empty() && val_records.is_empty() {
            return Err(anyhow::anyhow!("sudoku dataset contains no records"));
        }

        let mut train_records = train_records;
        let train_len = if val_records.is_empty() {
            let total = train_records.len();
            if total <= 1 {
                total
            } else {
                let mut split = ((total as f32) * config.train_split_ratio) as usize;
                split = split.clamp(1, total - 1);
                let tail = train_records.split_off(split);
                val_records = tail;
                split
            }
        } else {
            train_records.len()
        };

        let mut all_records = Vec::with_capacity(train_records.len() + val_records.len());
        all_records.extend(train_records);
        all_records.extend(val_records);

        let puzzles: Vec<Vec<u8>> =
            all_records.iter().map(|record| record.puzzle.clone()).collect();
        let solutions: Vec<Vec<u8>> =
            all_records.iter().map(|record| record.solution.clone()).collect();

        let summary = format!(
            "Prepared Sudoku dataset with batch_size={}, records={}, train_len={} (split_ratio={})",
            batch_size,
            puzzles.len(),
            train_len,
            config.train_split_ratio
        );

        Ok((
            Self {
                puzzles,
                solutions,
                train_len,
                batch_size,
                train_split_ratio: config.train_split_ratio,
            },
            summary,
        ))
    }

    pub fn len(&self) -> usize {
        self.puzzles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.puzzles.is_empty()
    }

    pub fn train_len(&self) -> usize {
        self.train_len
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }

    fn split_offset_and_span(&self, split: SudokuSplit) -> (usize, usize) {
        match split {
            SudokuSplit::Train => (0, self.train_len),
            SudokuSplit::Val => {
                let total = self.puzzles.len();
                let remaining = total.saturating_sub(self.train_len);
                if remaining == 0 {
                    (0, self.train_len)
                } else {
                    (self.train_len, remaining)
                }
            }
        }
    }

    pub fn steps_per_epoch(&self, split: SudokuSplit) -> usize {
        let (_offset, span) = self.split_offset_and_span(split);
        if self.batch_size == 0 {
            return 1;
        }
        span.div_ceil(self.batch_size).max(1)
    }

    pub fn sample_batch<B: Backend>(&self, split: SudokuSplit, device: &B::Device) -> SudokuBatch<B> {
        let (offset, span) = self.split_offset_and_span(split);
        let mut rng = thread_rng();
        let mut puzzles = vec![0i64; self.batch_size * GRID_LEN];
        let mut solutions = vec![0i64; self.batch_size * GRID_LEN];

        for batch_idx in 0..self.batch_size {
            let idx = if span == 0 {
                0
            } else {
                offset + rng.gen_range(0..span)
            };
            let puzzle = &self.puzzles[idx];
            let solution = &self.solutions[idx];
            for cell in 0..GRID_LEN {
                puzzles[batch_idx * GRID_LEN + cell] = puzzle[cell] as i64;
                solutions[batch_idx * GRID_LEN + cell] = solution[cell] as i64;
            }
        }

        let puzzles_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(puzzles, [self.batch_size, GRID_LEN]),
            device,
        );
        let solutions_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(solutions, [self.batch_size, GRID_LEN]),
            device,
        );

        SudokuBatch::new(puzzles_tensor, solutions_tensor)
    }
}

pub struct SudokuRandomDataLoader<B: Backend> {
    dataset: Arc<SudokuDataset>,
    split: SudokuSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Clone for SudokuRandomDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
        }
    }
}

impl<B: Backend> SudokuRandomDataLoader<B> {
    pub fn new(
        dataset: Arc<SudokuDataset>,
        split: SudokuSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
    ) -> Self {
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch,
            total_steps,
            consumed_steps,
        }
    }
}

impl<B> DataLoader<B, SudokuBatch<B>> for SudokuRandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SudokuBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        Box::new(RandomIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch * self.dataset.batch_size()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SudokuBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SudokuBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: steps,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
        })
    }
}

struct RandomIterator<B: Backend> {
    dataset: Arc<SudokuDataset>,
    split: SudokuSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Iterator for RandomIterator<B> {
    type Item = SudokuBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        Some(self.dataset.sample_batch::<B>(self.split, &self.device))
    }
}

impl<B: Backend> DataLoaderIterator<SudokuBatch<B>> for RandomIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.dataset.batch_size(),
            self.steps_total * self.dataset.batch_size(),
        )
    }
}

fn load_hf_records(
    cfg: &SudokuHuggingFaceConfig,
    cache_dir: &Path,
) -> Result<(Vec<SudokuRecord>, Vec<SudokuRecord>), anyhow::Error> {
    let hf_cache_dir = cache_dir.join("huggingface");
    fs::create_dir_all(&hf_cache_dir)?;

    let token = cfg
        .token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .filter(|value| !value.trim().is_empty());

    let mut api_builder = ApiBuilder::new().with_cache_dir(hf_cache_dir);
    if let Some(token) = token {
        api_builder = api_builder.with_token(Some(token));
    }
    let api = api_builder.build().map_err(io::Error::other)?;

    let repo = if let Some(revision) = &cfg.revision {
        Repo::with_revision(cfg.repo_id.clone(), RepoType::Dataset, revision.clone())
    } else {
        Repo::new(cfg.repo_id.clone(), RepoType::Dataset)
    };
    let repo = api.repo(repo);

    let mut train_records = Vec::new();
    for file in &cfg.train_files {
        if cfg
            .max_records
            .is_some_and(|limit| train_records.len() >= limit)
        {
            break;
        }
        let path = repo
            .get(file)
            .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
        collect_records(
            &path,
            cfg.format.clone(),
            &cfg.puzzle_field,
            &cfg.solution_field,
            cfg.max_records,
            &mut train_records,
        )?;
    }

    let mut val_records = Vec::new();
    for file in &cfg.validation_files {
        let path = repo
            .get(file)
            .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
        collect_records(
            &path,
            cfg.format.clone(),
            &cfg.puzzle_field,
            &cfg.solution_field,
            cfg.max_records,
            &mut val_records,
        )?;
    }

    Ok((train_records, val_records))
}

fn load_local_records(
    cfg: &SudokuLocalConfig,
) -> Result<(Vec<SudokuRecord>, Vec<SudokuRecord>), anyhow::Error> {
    let mut train_records = Vec::new();
    for file in &cfg.train_files {
        if cfg
            .max_records
            .is_some_and(|limit| train_records.len() >= limit)
        {
            break;
        }
        let path = cfg.root.join(file);
        collect_records(
            &path,
            cfg.format.clone(),
            &cfg.puzzle_field,
            &cfg.solution_field,
            cfg.max_records,
            &mut train_records,
        )?;
    }

    let mut val_records = Vec::new();
    for file in &cfg.validation_files {
        let path = cfg.root.join(file);
        collect_records(
            &path,
            cfg.format.clone(),
            &cfg.puzzle_field,
            &cfg.solution_field,
            cfg.max_records,
            &mut val_records,
        )?;
    }

    Ok((train_records, val_records))
}

fn collect_records(
    path: &Path,
    format: SudokuRecordFormat,
    puzzle_field: &str,
    solution_field: &str,
    max_records: Option<usize>,
    records: &mut Vec<SudokuRecord>,
) -> io::Result<()> {
    match format {
        SudokuRecordFormat::Jsonl => collect_jsonl_records(path, puzzle_field, solution_field, max_records, records),
        SudokuRecordFormat::Csv => collect_csv_records(path, puzzle_field, solution_field, max_records, records),
        SudokuRecordFormat::Parquet => collect_parquet_records(path, puzzle_field, solution_field, max_records, records),
    }
}

fn collect_jsonl_records(
    path: &Path,
    puzzle_field: &str,
    solution_field: &str,
    max_records: Option<usize>,
    records: &mut Vec<SudokuRecord>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse JSON record from {}: {err}", path.display()),
            )
        })?;

        let puzzle = value.get(puzzle_field).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing `{puzzle_field}` in dataset record"),
            )
        })?;
        let solution = value.get(solution_field).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing `{solution_field}` in dataset record"),
            )
        })?;

        if let (Some(puzzle), Some(solution)) = (puzzle.as_str(), solution.as_str())
            && let Ok(record) = parse_record(puzzle, solution)
        {
            records.push(record);
        }
    }

    Ok(())
}

fn collect_csv_records(
    path: &Path,
    puzzle_field: &str,
    solution_field: &str,
    max_records: Option<usize>,
    records: &mut Vec<SudokuRecord>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let headers = reader.headers().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to read CSV headers from {}: {err}", path.display()),
        )
    })?;

    let puzzle_idx = headers.iter().position(|h| h == puzzle_field).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing field `{}` in csv file {}", puzzle_field, path.display()),
        )
    })?;
    let solution_idx = headers.iter().position(|h| h == solution_field).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing field `{}` in csv file {}", solution_field, path.display()),
        )
    })?;

    for record in reader.records() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let record = record.map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to read CSV record from {}: {err}", path.display()),
            )
        })?;
        if record.is_empty() {
            continue;
        }
        let puzzle = record.get(puzzle_idx).unwrap_or("");
        let solution = record.get(solution_idx).unwrap_or("");
        if let Ok(record) = parse_record(puzzle, solution) {
            records.push(record);
        }
    }

    Ok(())
}

fn collect_parquet_records(
    path: &Path,
    puzzle_field: &str,
    solution_field: &str,
    max_records: Option<usize>,
    records: &mut Vec<SudokuRecord>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = SerializedFileReader::new(file).map_err(io::Error::other)?;
    let schema = reader.metadata().file_metadata().schema_descr();

    let mut index_map = std::collections::HashMap::new();
    for (idx, column) in schema.columns().iter().enumerate() {
        index_map.insert(column.path().string(), idx);
    }

    let puzzle_idx = *index_map.get(puzzle_field).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing field `{}` in parquet file {}", puzzle_field, path.display()),
        )
    })?;
    let solution_idx = *index_map.get(solution_field).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("missing field `{}` in parquet file {}", solution_field, path.display()),
        )
    })?;

    let row_iter = reader.get_row_iter(None).map_err(io::Error::other)?;
    for row in row_iter {
        let row = row.map_err(io::Error::other)?;
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }

        let puzzle = row
            .get_string(puzzle_idx)
            .map(|s| s.to_string())
            .or_else(|_| row.get_bytes(puzzle_idx).map(|bytes| String::from_utf8_lossy(bytes.data()).to_string()))
            .or_else(|_| {
                row.get_column_iter()
                    .nth(puzzle_idx)
                    .map(|(_, field)| field.to_string())
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing puzzle"))
            })?;

        let solution = row
            .get_string(solution_idx)
            .map(|s| s.to_string())
            .or_else(|_| row.get_bytes(solution_idx).map(|bytes| String::from_utf8_lossy(bytes.data()).to_string()))
            .or_else(|_| {
                row.get_column_iter()
                    .nth(solution_idx)
                    .map(|(_, field)| field.to_string())
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing solution"))
            })?;

        if let Ok(record) = parse_record(&puzzle, &solution) {
            records.push(record);
        }
    }

    Ok(())
}

fn parse_record(puzzle: &str, solution: &str) -> Result<SudokuRecord, anyhow::Error> {
    let puzzle = SudokuVocab::encode_grid(puzzle)?;
    let solution = SudokuVocab::encode_grid(solution)?;
    if puzzle.len() != GRID_LEN || solution.len() != GRID_LEN {
        return Err(anyhow::anyhow!("invalid grid length"));
    }
    Ok(SudokuRecord { puzzle, solution })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn local_jsonl_dataset_loads() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("train.jsonl");
        let payload = [
            r#"{"puzzle":"530070000600195000098000060800060003400803001700020006060000280000419005000080079","solution":"534678912672195348198342567859761423426853791713924856961537284287419635345286179"}"#,
            r#"{"puzzle":"003020600900305001001806400008102900700000008006708200002609500800203009005010300","solution":"483921657967345821251876493548132976729564138136798245372689514814253769695417382"}"#,
        ]
        .join("\n");
        fs::write(&path, payload).expect("write dataset");

        let cfg = SudokuLocalConfig {
            root: dir.path().to_path_buf(),
            format: SudokuRecordFormat::Jsonl,
            train_files: vec!["train.jsonl".to_string()],
            validation_files: Vec::new(),
            puzzle_field: "puzzle".to_string(),
            solution_field: "solution".to_string(),
            max_records: None,
        };
        let dataset_cfg = SudokuDatasetConfig {
            cache_dir: dir.path().to_path_buf(),
            train_split_ratio: 0.5,
            source: SudokuDatasetSourceConfig::Local(cfg),
        };

        let (dataset, _summary) = SudokuDataset::new(&dataset_cfg, 2).expect("dataset");
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.train_len(), 1);
    }
}
