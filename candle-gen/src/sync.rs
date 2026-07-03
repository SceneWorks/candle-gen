//! Poison-tolerant `Mutex` locking for the shared generator/component caches (sc-9015 / F-031).
//!
//! Several provider crates keep a lazily-populated, overwrite-on-miss cache — the loaded
//! UNet/VAE `Components` behind a `Mutex<Option<…>>`, or a streaming-conv `feat_cache` behind a
//! `Mutex<Option<Tensor>>`. Each `generate` runs on a shared `Arc<dyn Generator>` (`&self`), so a
//! panic *while holding the lock* (e.g. a CUDA OOM lifted to a panic mid-decode) poisons the
//! `Mutex`. With a plain `.lock().unwrap()` / `.expect(…)`, every subsequent lock then panics
//! forever — one transient failure wedges a long-lived worker lane into a permanent panic loop
//! until the process restarts.
//!
//! These caches carry no cross-field invariant that a mid-op panic can leave half-broken: the state
//! is a single `Option` that is *unconditionally overwritten on the next miss* (or reset). A
//! partially observed value is at worst re-loaded/re-computed. So the correct recovery is to treat a
//! poisoned lock as usable and keep serving, rather than propagating the poison. [`lock_recover`]
//! does exactly `lock().unwrap_or_else(|e| e.into_inner())`.
//!
//! Do NOT route a lock through this helper if the guarded data has a multi-field invariant that a
//! panic mid-mutation could violate — recovering such a lock would hand out inconsistent state.

use std::sync::{Mutex, MutexGuard};

/// Lock `m`, recovering from poisoning by taking the inner guard.
///
/// A poisoned `Mutex` (a prior holder panicked) is treated as usable: the guarded value is an
/// overwrite-on-miss cache with no invariant a partial write can break (see the module docs), so we
/// keep serving instead of turning one panic into permanent panics on the shared generator cache
/// (sc-9015 / F-031). Equivalent to `m.lock().unwrap_or_else(|e| e.into_inner())`.
#[inline]
pub fn lock_recover<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::lock_recover;
    use std::sync::{Arc, Mutex};

    #[test]
    fn recovers_from_a_poisoned_cache_mutex() {
        // Model the shared generator cache: a `Mutex<Option<…>>` populated once, overwrite-on-miss.
        let cache: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(Some(7)));

        // Poison it exactly like a panic-while-locked mid-decode would: a spawned thread grabs the
        // lock and unwinds while holding the guard.
        let poisoner = Arc::clone(&cache);
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.lock().expect("first lock is clean");
            panic!("simulated CUDA OOM mid-decode while holding the cache lock");
        });
        assert!(
            handle.join().is_err(),
            "the poisoning thread must have panicked"
        );
        assert!(cache.is_poisoned(), "the mutex must now be poisoned");

        // A plain `.lock().unwrap()` here would panic forever. The recovery path keeps serving the
        // cached value...
        assert_eq!(
            *lock_recover(&cache),
            Some(7),
            "poisoned cache still readable"
        );

        // ...and stays writable (overwrite-on-miss is still honored after recovery).
        *lock_recover(&cache) = Some(42);
        assert_eq!(
            *lock_recover(&cache),
            Some(42),
            "poisoned cache still writable"
        );
    }
}
