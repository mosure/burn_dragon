use std::sync::Arc;

pub trait StreamDataset {
    type Item;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::Item>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug)]
pub struct InMemoryStreamDataset<T> {
    items: Arc<Vec<T>>,
}

impl<T> InMemoryStreamDataset<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self {
            items: Arc::new(items),
        }
    }
}

impl<T: Clone> StreamDataset for InMemoryStreamDataset<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.items.get(index).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::{InMemoryStreamDataset, StreamDataset};

    #[test]
    fn in_memory_dataset_supports_dummy_payloads() {
        let dataset = InMemoryStreamDataset::new(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.get(1).as_deref(), Some("b"));
        assert_eq!(dataset.get(2), None);
    }
}
