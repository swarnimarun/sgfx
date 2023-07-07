use std::{
    future::Future,
    time::{Duration, Instant, SystemTime},
};

use anyhow::Result;
use log;
use wgpu::{Backend, SurfaceError};
use winit::{
    event::{Event, KeyboardInput},
    event_loop::{self, ControlFlow},
};

pub trait GfxApp {
    type EventState: 'static;
    fn backend() -> Backend {
        Backend::Vulkan
    }
    fn init(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self;
    fn handle_event(&mut self, ev: &Event<Self::EventState>, cf: &mut ControlFlow) {
        match ev {
            Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode, ..
                        },
                    ..
                } => match virtual_keycode {
                    Some(winit::event::VirtualKeyCode::Escape) => *cf = ControlFlow::Exit,
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }
    }
    fn render(&mut self, frame: &wgpu::SurfaceTexture, device: &wgpu::Device, queue: &wgpu::Queue);
    fn update(&mut self, delta_time: f64) {}
    fn exit(&mut self) {}
}

/// Window for sgfx application
pub struct Window<App: GfxApp + 'static> {
    app: App,
    event_loop: winit::event_loop::EventLoop<App::EventState>,
    size: WindowSize,
    title: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    window: winit::window::Window,
}
impl<App: GfxApp + 'static> Window<App> {
    pub fn run(self) -> Result<()> {
        let Self {
            mut app,
            event_loop,
            window,
            surface,
            device,
            mut config,
            queue,
            ..
        } = self;
        // limit framerate
        let mut then = SystemTime::now();
        let mut now = SystemTime::now();
        let mut frames = 0;
        // change '60.0' if you want different FPS cap
        let target_framerate = Duration::from_secs_f64(1.0 / 60.0);
        let mut delta_time = Instant::now();
        let mut fps = 0.0;
        // ---------------
        event_loop.run(move |ev, _window_target, cf| {
            *cf = ControlFlow::Poll;
            app.handle_event(&ev, cf);
            match ev {
                Event::WindowEvent { event, .. } => match event {
                    winit::event::WindowEvent::CloseRequested => {
                        *cf = ControlFlow::Exit;
                    }
                    winit::event::WindowEvent::Resized(size) => {
                        config.width = size.width.max(1);
                        config.height = size.height.max(1);
                        surface.configure(&device, &config);
                    }
                    _ => {}
                },
                Event::RedrawRequested(_) => {
                    let frame = match surface.get_current_texture() {
                        Ok(st) => st,
                        Err(_) => {
                            surface.configure(&device, &config);
                            surface
                                .get_current_texture()
                                .expect("failed to get frame/surface texture")
                        }
                    };
                    app.render(&frame, &device, &queue);
                    frame.present();
                    frames += 1;
                    if let Ok(dur) = now.duration_since(then) {
                        if dur.as_millis() > 100 {
                            fps = (frames as f64 / (dur.as_millis() as f64 / 100.0)) * 10.0;
                            frames = 0;
                            then = now;
                        }
                    }
                    now = SystemTime::now();
                }
                Event::MainEventsCleared => {
                    if target_framerate <= delta_time.elapsed() {
                        window.request_redraw();
                        delta_time = Instant::now();
                    } else {
                        *cf = ControlFlow::WaitUntil(
                            Instant::now().checked_sub(delta_time.elapsed()).unwrap()
                                + target_framerate,
                        );
                    }
                    app.update(delta_time.elapsed().as_secs_f64());
                }
                Event::LoopDestroyed => {
                    app.exit();
                }
                _ => {}
            }
        })
    }
}

/// Only for initializing a window
#[derive(Debug)]
pub enum WindowSize {
    /// Custom window size, rendered as windowed
    Windowed(usize, usize),
    /// Will change screen aspect ratio, and resolution
    Fullscreen(usize, usize),
    /// Will be rendered on a surface of the size of the monitor,
    /// but it's true fullscreen useful as it simplifies rendering costs.
    BorderlessWindow,
}

#[derive(Debug, Default)]
pub struct WindowBuilder {
    title: Option<String>,
    size: Option<WindowSize>,
}

impl WindowBuilder {
    pub fn build<App: GfxApp>(self) -> Result<Window<App>> {
        let event_loop = event_loop::EventLoopBuilder::with_user_event().build();
        let Self { title, size } = self;
        let title = title.unwrap_or_else(|| {
            log::warn!("Title not set, using \"Default Window Title\"");
            "Default Window Title".to_string()
        });
        let size = size.unwrap_or_else(|| {
            log::warn!("Size not set, using default window size (1024, 768)");
            WindowSize::Windowed(1024, 768)
        });
        let window = winit::window::WindowBuilder::new().build(&event_loop)?;
        let (device, queue, surface, config) = poll_spin(wgpu_init(&window));
        Ok(Window {
            app: App::init(&device, &config),
            event_loop,
            size,
            title,
            device,
            queue,
            surface,
            config,
            window,
        })
    }
    pub fn set_size(mut self, size: WindowSize) -> Self {
        _ = self.size.insert(size);
        self
    }
    pub fn set_title(mut self, title: String) -> Self {
        _ = self.title.insert(title);
        self
    }
}

pub async fn wgpu_init(
    window: &winit::window::Window,
) -> (
    wgpu::Device,
    wgpu::Queue,
    wgpu::Surface,
    wgpu::SurfaceConfiguration,
) {
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    // instance of wgpu represents the wgpu interface b/w wgpu api, graphics api and GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        // TODO(swarnim): do not use Fxc
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
    });
    let surface = unsafe { instance.create_surface(window) }
        .expect("failed to create a surface from instance");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: Default::default(),
            },
            None,
        )
        .await
        .unwrap();
    let caps = surface.get_capabilities(&adapter);
    let format = *caps.formats.iter().find(|x| x.is_srgb()).unwrap();
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoNoVsync, // *caps.present_modes.get(0).unwrap(),
        alpha_mode: *caps.alpha_modes.get(0).unwrap(),
        view_formats: vec![],
    };
    surface.configure(&device, &config);
    (device, queue, surface, config)
}

// TODO(swarnim): actually write this in a more production useful way
// dumbest polling of a future
pub fn poll_spin<T>(f: impl Future<Output = T>) -> T {
    struct S;
    impl std::task::Wake for S {
        fn wake(self: std::sync::Arc<Self>) {
            // do nothing tbh
        }
    }
    let mut f = Box::pin(f);
    let waker = std::task::Waker::from(std::sync::Arc::new(S));
    let mut ctx = std::task::Context::from_waker(&waker);
    // this is practically going to burn cycles to poll as the waker is redudandant
    loop {
        match f.as_mut().poll(&mut ctx) {
            std::task::Poll::Ready(v) => break v,
            std::task::Poll::Pending => {}
        }
    }
}
