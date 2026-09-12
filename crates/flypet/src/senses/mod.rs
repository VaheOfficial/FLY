//! How the desktop reaches the fly's sense organs.
//!
//! Each desktop signal is turned into a Poisson rate onto a named sensory
//! population. The mapping constants here are the only place where "the
//! user typed" becomes "the substrate vibrated"; the brain crate never
//! knows about keyboards.

pub mod keyboard;

use std::time::Duration;

use anyhow::{Context, Result};
use keyboard::KeyboardHook;

/// One keystroke's instantaneous drive onto the leg chordotonal organs, Hz.
const SUBSTRATE_HZ_PER_KEY: f32 = 150.0;
/// How fast a single tap's vibration dies away.
const SUBSTRATE_DECAY: Duration = Duration::from_millis(100);
/// Ceiling for substrate drive; frantic typing saturates the sensors.
const SUBSTRATE_MAX_HZ: f32 = 400.0;

/// One keystroke's contribution to what Johnston's organ hears as song, Hz.
/// Small per key and slow to fade, so only sustained rhythmic typing builds
/// up to a real song.
const SONG_HZ_PER_KEY: f32 = 20.0;
const SONG_DECAY: Duration = Duration::from_millis(1500);
const SONG_MAX_HZ: f32 = 300.0;

/// Sensory drive for one frame, as Poisson rates onto each organ.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vibration {
    /// Onto the leg chordotonal organs: the surface shaking under the fly.
    pub substrate_hz: f32,
    /// Onto Johnston's organ sound neurons: rhythm heard as courtship song.
    pub song_hz: f32,
}

pub struct Senses {
    keyboard: KeyboardHook,
    substrate_energy: f32,
    song_energy: f32,
}

impl Senses {
    pub fn install() -> Result<Self> {
        Ok(Self {
            keyboard: KeyboardHook::install().context("installing keyboard hook")?,
            substrate_energy: 0.0,
            song_energy: 0.0,
        })
    }

    /// Fold `elapsed` of desktop activity into sensory drive.
    pub fn update(&mut self, elapsed: Duration) -> Vibration {
        let keys = self.keyboard.take_keystrokes() as f32;
        self.substrate_energy =
            decay(self.substrate_energy, elapsed, SUBSTRATE_DECAY) + keys * SUBSTRATE_HZ_PER_KEY;
        self.song_energy = decay(self.song_energy, elapsed, SONG_DECAY) + keys * SONG_HZ_PER_KEY;
        Vibration {
            substrate_hz: self.substrate_energy.min(SUBSTRATE_MAX_HZ),
            song_hz: self.song_energy.min(SONG_MAX_HZ),
        }
    }
}

fn decay(value: f32, elapsed: Duration, time_constant: Duration) -> f32 {
    value * (-elapsed.as_secs_f32() / time_constant.as_secs_f32()).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_halves_at_ln2_time_constants() {
        let half_life = Duration::from_secs_f32(SUBSTRATE_DECAY.as_secs_f32() * 2f32.ln());
        assert!((decay(100.0, half_life, SUBSTRATE_DECAY) - 50.0).abs() < 1e-3);
        assert_eq!(decay(100.0, Duration::ZERO, SUBSTRATE_DECAY), 100.0);
    }
}
