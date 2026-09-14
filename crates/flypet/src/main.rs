//! `flypet`: the desktop shell. Translates desktop state into sensory input for
//! `flybrain` and renders the fly.
//!
//! Current state: a small transparent, click-through, always-on-top pet
//! window whose opacity follows courtship arousal, and a brain viewer window
//! showing every neuron light up as it spikes. Keystrokes anywhere on the
//! desktop shake the surface under the fly and, when sustained, reach its
//! ear.

mod brain;
mod gpu;
mod render;
mod senses;
mod snapshot;
mod viewer;

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use flybrain::connectome::Connectome;
use flybrain::morphology::Morphology;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId, WindowLevel};

use brain::Brain;
use gpu::Gpu;
use render::Renderer;
use senses::Senses;
use viewer::Viewer;

const FRAME: Duration = Duration::from_micros(16_667);
const PET_SIZE: u32 = 160;
const VIEWER_SIZE: (u32, u32) = (1100, 640);
/// Opacity reaches full at this pC1 rate.
const FULL_OPACITY_HZ: f32 = 50.0;

#[derive(Parser)]
#[command(about)]
struct Args {
    /// Packed connectome file.
    #[arg(default_value = "data/malecns-v1.0.flycnx")]
    connectome: PathBuf,
    /// Packed morphology file for the viewer; skipped if absent.
    #[arg(long, default_value = "data/malecns-v1.0.flyskel")]
    morphology: PathBuf,
    /// Save the brain viewer to this PNG after `--snapshot-after` seconds, then exit.
    #[arg(long)]
    snapshot: Option<PathBuf>,
    #[arg(long, default_value_t = 3.0)]
    snapshot_after: f32,
    /// Drive the sugar GRNs at this rate so the viewer has activity to show.
    #[arg(long, default_value_t = 0.0)]
    sugar_hz: f32,
    /// Draw morphology only, without cell bodies.
    #[arg(long)]
    hide_somas: bool,
}

struct App<'a> {
    args: &'a Args,
    net: &'a Connectome,
    morphology: Option<&'a Morphology>,
    gpu: Option<Gpu>,
    pet_window: Option<Arc<Window>>,
    viewer_window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    viewer: Option<Viewer>,
    brain: Option<Brain<'a>>,
    senses: Option<Senses>,
    started: Instant,
    last_tick: Instant,
    next_frame: Instant,
    last_report: Instant,
}

impl<'a> App<'a> {
    fn new(args: &'a Args, net: &'a Connectome, morphology: Option<&'a Morphology>) -> Self {
        let now = Instant::now();
        Self {
            args,
            net,
            morphology,
            gpu: None,
            pet_window: None,
            viewer_window: None,
            renderer: None,
            viewer: None,
            brain: None,
            senses: None,
            started: now,
            last_tick: now,
            next_frame: now,
            last_report: now,
        }
    }

    fn open_windows(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let pet = Window::default_attributes()
            .with_title("FLY")
            .with_inner_size(PhysicalSize::new(PET_SIZE, PET_SIZE))
            .with_position(PhysicalPosition::new(80, 80))
            .with_transparent(true)
            .with_decorations(false)
            .with_window_level(WindowLevel::AlwaysOnTop);
        let pet_window = Arc::new(event_loop.create_window(pet)?);
        if let Err(e) = pet_window.set_cursor_hittest(false) {
            eprintln!("click-through unavailable: {e}");
        }
        let viewer_attributes = Window::default_attributes()
            .with_title("FLY brain")
            .with_inner_size(PhysicalSize::new(VIEWER_SIZE.0, VIEWER_SIZE.1))
            .with_position(PhysicalPosition::new(300, 80));
        let viewer_window = Arc::new(event_loop.create_window(viewer_attributes)?);

        let (gpu, pet_surface) = Gpu::open(&pet_window, self.net)?;
        let pet_size = pet_window.inner_size();
        let renderer = Renderer::new(&gpu, pet_surface, (pet_size.width, pet_size.height))?;
        let viewer_surface = gpu.surface(&viewer_window)?;
        let viewer_size = viewer_window.inner_size();
        let mut viewer = Viewer::new(
            &gpu,
            viewer_surface,
            (viewer_size.width, viewer_size.height),
            self.net,
            self.morphology,
        )?;
        viewer.show_somas = !self.args.hide_somas;
        println!("viewer: {}", viewer.describe());

        let mut brain = Brain::new(self.net, Some(gpu.sim_context()));
        brain.set_sugar_hz(self.args.sugar_hz);
        println!("backend: {}", brain.backend_name());
        println!(
            "populations: leg CO {}, JO sound {}, pC1 {}",
            brain.population_size(brain.leg_chordotonal),
            brain.population_size(brain.johnstons_organ),
            brain.population_size(brain.courtship_arousal)
        );
        let senses = Senses::install()?;

        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.viewer = Some(viewer);
        self.brain = Some(brain);
        self.senses = Some(senses);
        self.pet_window = Some(pet_window);
        self.viewer_window = Some(viewer_window);
        self.started = Instant::now();
        self.last_tick = self.started;
        Ok(())
    }

    /// One frame: sense, tick, draw both windows. Returns true once a
    /// requested snapshot has been written.
    fn frame(&mut self) -> Result<bool> {
        let (Some(gpu), Some(brain), Some(renderer), Some(viewer), Some(senses)) = (
            self.gpu.as_ref(),
            self.brain.as_mut(),
            self.renderer.as_mut(),
            self.viewer.as_mut(),
            self.senses.as_mut(),
        ) else {
            return Ok(false);
        };
        let now = Instant::now();
        let elapsed = now - self.last_tick;
        self.last_tick = now;

        let vibration = senses.update(elapsed);
        brain.feel(vibration);
        brain.tick(elapsed);
        viewer.update(brain.last_spike_counts(), elapsed);

        let arousal_hz = brain.rate_hz(brain.courtship_arousal);
        let opacity = f64::from((arousal_hz / FULL_OPACITY_HZ).clamp(0.15, 1.0));
        renderer.present(gpu, [0.95, 0.55, 0.1], opacity)?;

        let snapshot_due =
            self.args.snapshot.as_deref().filter(|_| {
                now.duration_since(self.started).as_secs_f32() >= self.args.snapshot_after
            });
        viewer.draw(gpu, snapshot_due)?;

        if now.duration_since(self.last_report) >= Duration::from_secs(1) {
            self.last_report = now;
            // A closed console must not take the pet down with it.
            let _ = writeln!(
                std::io::stdout(),
                "substrate {:5.0} Hz  song {:5.0} Hz | leg CO {:6.1} Hz  JO {:6.1} Hz  pC1 {:6.1} Hz",
                vibration.substrate_hz,
                vibration.song_hz,
                brain.rate_hz(brain.leg_chordotonal),
                brain.rate_hz(brain.johnstons_organ),
                arousal_hz
            );
        }
        Ok(snapshot_due.is_some())
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.pet_window.is_some() {
            return;
        }
        if let Err(e) = self.open_windows(event_loop) {
            eprintln!("startup failed: {e:#}");
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let is_viewer = self.viewer_window.as_ref().is_some_and(|w| w.id() == id);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_ref() {
                    if is_viewer {
                        if let Some(viewer) = self.viewer.as_mut() {
                            viewer.resize(gpu, size.width, size.height);
                        }
                    } else if let Some(renderer) = self.renderer.as_mut() {
                        renderer.resize(gpu, size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested if !is_viewer => match self.frame() {
                Ok(true) => {
                    println!("snapshot written, exiting");
                    event_loop.exit();
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!("frame failed: {e:#}");
                    event_loop.exit();
                }
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if now >= self.next_frame {
            self.next_frame = now + FRAME;
            if let Some(window) = &self.pet_window {
                window.request_redraw();
            }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let net = brain::load_connectome(&args.connectome)?;
    println!(
        "loaded {} neurons, {} edges",
        net.neuron_count(),
        net.edge_count()
    );

    let morphology = brain::load_morphology(&args.morphology)?;
    match &morphology {
        Some(m) => println!(
            "loaded morphology: {} vertices in {} strips",
            m.vertex_count(),
            m.strips.len()
        ),
        None => println!(
            "no morphology file at {}; drawing somas only",
            args.morphology.display()
        ),
    }

    let event_loop = EventLoop::new()?;
    let mut app = App::new(&args, &net, morphology.as_ref());
    event_loop.run_app(&mut app)?;
    Ok(())
}
