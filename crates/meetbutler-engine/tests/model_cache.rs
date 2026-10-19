//! Models are expensive to load and safe to share: one per configuration per process,
//! never one per session.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use meetbutler_engine::model_cache::ModelCache;

#[derive(Debug)]
struct Model(String);

fn counting(loads: &Arc<AtomicUsize>) -> impl Fn(&String) -> Result<Model, String> + '_ {
    move |key: &String| {
        loads.fetch_add(1, Ordering::SeqCst);
        Ok(Model(key.clone()))
    }
}

#[test]
fn same_config_reuses_one_loaded_model() {
    let cache = ModelCache::new();
    let loads = Arc::new(AtomicUsize::new(0));

    let a = cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();
    let b = cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();

    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}

#[test]
fn different_config_loads_a_separate_model() {
    let cache = ModelCache::new();
    let loads = Arc::new(AtomicUsize::new(0));

    let small = cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();
    let turbo = cache
        .get_or_load("turbo".to_string(), counting(&loads))
        .unwrap();

    assert_eq!((small.0.as_str(), turbo.0.as_str()), ("small", "turbo"));
    assert_eq!(loads.load(Ordering::SeqCst), 2);
}

#[test]
fn many_sessions_share_a_single_model_even_when_they_ask_at_once() {
    let cache = Arc::new(ModelCache::new());
    let loads = Arc::new(AtomicUsize::new(0));

    let threads: Vec<_> = (0..8)
        .map(|_| {
            let (cache, loads) = (cache.clone(), loads.clone());
            std::thread::spawn(move || {
                cache
                    .get_or_load("small".to_string(), |k: &String| {
                        loads.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        Ok::<_, String>(Model(k.clone()))
                    })
                    .unwrap()
            })
        })
        .collect();
    let models: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();

    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert!(models.windows(2).all(|w| Arc::ptr_eq(&w[0], &w[1])));
}

#[test]
fn a_failed_load_is_not_cached() {
    let cache: ModelCache<String, Model> = ModelCache::new();
    assert!(cache
        .get_or_load("small".to_string(), |_: &String| Err("no file"))
        .is_err());

    let loads = Arc::new(AtomicUsize::new(0));
    assert!(cache
        .get_or_load("small".to_string(), counting(&loads))
        .is_ok());
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}

#[test]
fn evicting_releases_the_model_for_the_next_load() {
    // the final pass unloads the live model before loading the large one (memory budget)
    let cache = ModelCache::new();
    let loads = Arc::new(AtomicUsize::new(0));
    cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();

    cache.evict(&"small".to_string());
    cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();

    assert_eq!(loads.load(Ordering::SeqCst), 2);
}

#[test]
fn clear_drops_every_model() {
    let cache = ModelCache::new();
    let loads = Arc::new(AtomicUsize::new(0));
    cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();
    cache
        .get_or_load("turbo".to_string(), counting(&loads))
        .unwrap();

    cache.clear();
    cache
        .get_or_load("small".to_string(), counting(&loads))
        .unwrap();

    assert_eq!(loads.load(Ordering::SeqCst), 3);
}
