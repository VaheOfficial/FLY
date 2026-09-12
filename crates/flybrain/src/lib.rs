//! `flybrain`: a leaky integrate-and-fire simulation of a whole *Drosophila*
//! central nervous system connectome.
//!
//! This crate is deliberately free of windowing and desktop concerns so it can
//! be reused by the desktop pet (`flypet`), a firewall daemon, or a robot.

pub mod backend;
pub mod connectome;
pub mod gpu;
pub mod sim;
