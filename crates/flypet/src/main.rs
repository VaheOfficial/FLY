//! `flypet`: the desktop shell. Translates desktop state into sensory input for
//! `flybrain` and renders the fly.
//!
//! Current state: a small transparent, click-through, always-on-top window
//! whose opacity follows the proboscis motor neuron rate while the sugar
//! GRNs are stimulated. It proves the brain ticks in real time inside a
//! desktop window on a GPU shared with the renderer.

mod brain;
mod render;

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use flybrain::connectome::Connectome;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId, WindowLevel};

use brain::Brain;
use render::Renderer;

const FRAME: Duration = Duration::from_micros(16_667);
const WINDOW_SIZE: u32 = 160;
/// Sugar on the labellum, for now always present so there is something to see.
const SUGAR_HZ: f32 = 100.0;
/// Opacity reaches full at this MN9 rate.
const FULL_OPACITY_HZ: f32 = 200.0;

struct App<'a> {
    net: &'a Connectome,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    brain: Option<Brain<'a>>,
    last_tick: Instant,
    next_frame: Instant,
    last_report: Instant,
}

impl<'a> App<'a> {
    fn new(net: &'a Connectome) -> Self {
        let now = Instant::now();
        Self {
            net,
            window: None,
            renderer: None,
            brain: None,
            last_tick: now,
            next_frame: now,
            last_report: now,
        }
    }

    fn frame(&mut self) -> Result<()> {
        let (Some(brain), Some(renderer)) = (self.brain.as_mut(), self.renderer.as_mut()) else {
            return Ok(());
        };
        let now = Instant::now();
        brain.tick(now - self.last_tick);
        self.last_tick = now;

        let mn9_hz = brain.rate_hz(brain.proboscis_motor);
        let opacity = f64::from((mn9_hz / FULL_OPACITY_HZ).clamp(0.15, 1.0));
        renderer.present([0.95, 0.55, 0.1], opacity)?;

        if now.duration_since(self.last_report) >= Duration::from_secs(1) {
            self.last_report = now;
            // A closed console must not take the pet down with it.
            let _ = writeln!(
                std::io::stdout(),
                "sugar GRNs {:6.1} Hz   MN9 {:6.1} Hz",
                brain.rate_hz(brain.sugar_grns),
                mn9_hz
            );
        }
        Ok(())
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title("FLY")
            .with_inner_size(PhysicalSize::new(WINDOW_SIZE, WINDOW_SIZE))
            .with_position(PhysicalPosition::new(80, 80))
            .with_transparent(true)
            .with_decorations(false)
            .with_window_level(WindowLevel::AlwaysOnTop);
        let window = Arc::new(event_loop.create_window(attributes).expect("create window"));
        if let Err(e) = window.set_cursor_hittest(false) {
            eprintln!("click-through unavailable: {e}");
        }

        let (renderer, gpu) = Renderer::new(window.clone(), self.net).expect("GPU for window");
        let mut brain = Brain::new(self.net, Some(gpu));
        brain.set_sugar_hz(SUGAR_HZ);
        println!("backend: {}", brain.backend_name());

        self.renderer = Some(renderer);
        self.brain = Some(brain);
        self.window = Some(window);
        self.last_tick = Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame() {
                    eprintln!("frame failed: {e}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if now >= self.next_frame {
            self.next_frame = now + FRAME;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/malecns-v1.0.flycnx"));
    let net = brain::load_connectome(&path)?;
    println!(
        "loaded {} neurons, {} edges",
        net.neuron_count(),
        net.edge_count()
    );

    let event_loop = EventLoop::new()?;
    let mut app = App::new(&net);
    event_loop.run_app(&mut app)?;
    Ok(())
}
