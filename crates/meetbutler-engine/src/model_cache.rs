//! Load each model once per process and share it.
//!
//! A whisper.cpp context is expensive to load and safe to share between sessions;
//! language and other decoding options are per call, so one model serves every session.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

pub struct ModelCache<K, M> {
    models: Mutex<HashMap<K, Arc<M>>>,
}

impl<K: Eq + Hash + Clone, M> Default for ModelCache<K, M> {
    fn default() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
        }
    }
}

impl<K: Eq + Hash + Clone, M> ModelCache<K, M> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The shared model for `key`, loading it at most once. The lock is held while
    /// loading, so callers asking at the same time wait for one load instead of each
    /// starting their own. A failed load is not remembered.
    pub fn get_or_load<E>(
        &self,
        key: K,
        load: impl FnOnce(&K) -> Result<M, E>,
    ) -> Result<Arc<M>, E> {
        let mut models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(model) = models.get(&key) {
            return Ok(model.clone());
        }
        let model = Arc::new(load(&key)?);
        models.insert(key, model.clone());
        Ok(model)
    }

    /// Forget `key`. The model is freed once the last session using it lets go.
    pub fn evict(&self, key: &K) {
        self.models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
    }

    pub fn clear(&self) {
        self.models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}
