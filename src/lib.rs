// CMSC427 Lab 03 — Transforms, Textures & Lights

// renders a checkerboard-textured cube lit by a configurable spotlight
// viewed through an orbiting camera with toggleable perspective/ortho projection

// Additions since Lab 02:
// Vertex normals — each vert stores a surface normal ([f32;3])
// shader uses them for lighting calcs

// Texture sampling — checkerboard Rgba8UnormSrgb texture uploaded at startup frag shader samples it with textureSample()

// Blinn-Phong lighting — ambient + diffuse (lambertian) + specular (phong), spotlight cone modulates with smoothstep()

// Depth testing — Depth32Float texture prevents back faces from overwriting front faces. without it the cube looks inside-out depending on draw order (guess how I know)
//
// Four bind groups — camera, model matrix, texture+sampler and light
//wgpu defaults max out at 4 bind groups, so don't ask for more ;-;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

pub mod camera;
pub mod gpu;
pub mod light;
pub mod mesh;

use std::sync::Arc;
use std::time::Instant;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};
use camera::{CameraUniform, build_camera_uniform};
use gpu::{GpuCtx, uniform_bgl, uniform_buf, uniform_bg};
use light::{build_light_uniform, PRESET_COUNT};
use mesh::{Vertex, make_checkerboard, make_cube, make_depth_texture, upload_texture};

// Model Uniform
// model matrix transforms verts from object space → world space
// here encodes cube rotation - we animate it so cube spins while camera orbits

// separate from camera's view_proj bcos if we baked the model transform into each vert we'd have to re-upload 
// the whole vert buffer each frame.

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Model { model: [[f32; 4]; 4] }

//  WGSL Shader + bind group layouts 
//  group(0) binding(0) — CameraUniform (vert + frag) — view_proj + eye_pos
//  group(1) binding(0) — Model matrix (vert only) — per-object transform
//  group(2) binding(0) — texture_2d (frag only) — checkerboard diffuse
//  group(2) binding(1) — sampler (frag only) — bilinear, repeat
//  group(3) binding(0) — LightUniform (frag only) — spotlight params
const SHADER: &str = r#"
// uniform struct decls - must match Rust structs (same field order, same padding)
struct Camera { view_proj: mat4x4<f32>, eye_pos: vec3<f32>, _pad: f32 };
struct Model  { model: mat4x4<f32> };
struct Light  {
    pos: vec3<f32>,  _p0: f32,
    color: vec3<f32>, intensity: f32,
    dir: vec3<f32>,  _p1: f32,
    cos_inner: f32, cos_outer: f32, _p2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera:    Camera;
@group(1) @binding(0) var<uniform> model_u:   Model;
@group(2) @binding(0) var          t_diffuse: texture_2d<f32>;
@group(2) @binding(1) var          s_diffuse: sampler;
@group(3) @binding(0) var<uniform> light:     Light;

// @location(N) must match Vertex::layout()'s shader_location values
struct VIn  {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>
};
struct VOut {
    @builtin(position) clip:         vec4<f32>,
    @location(0)       world_pos:    vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       uv:           vec2<f32>
};

@vertex
fn vs_main(in: VIn) -> VOut {
    var out: VOut;
    // object space → world space (w=1 for points, translations apply)
    let world        = model_u.model * vec4<f32>(in.position, 1.0);
    // world space → clip space via combined view-projection
    out.clip         = camera.view_proj * world;
    // pass world-space position to frag for lighting
    out.world_pos    = world.xyz;
    // transform normal by model (w=0, translations don't apply)
    // normalize bc model might have non-uniform scale
    out.world_normal = normalize((model_u.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv           = in.uv;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // sample checkerboard at UV coords
    let tex   = textureSample(t_diffuse, s_diffuse, in.uv);

    // re-normalize interpolated normal (interpolation can shrink it)
    let N     = normalize(in.world_normal);
    // L = direction from surface to light
    let L     = normalize(light.pos - in.world_pos);
    // V = direction from surface to eye (for specular half-vector)
    let V     = normalize(camera.eye_pos - in.world_pos);
    // H = blinn-phong half-vector (if surface is perfect mirror reflecting L toward V, N==H)
    let H     = normalize(L + V);

    // spotlight cone falloff
    // dot(-L, normalize(light.dir)) measures alignment btwn light-to-surface and spotlight dir
    // smoothstep returns 0 at cos_outer, 1 at cos_inner, smooth cubic blend btwn
    let spot = smoothstep(light.cos_outer, light.cos_inner, dot(-L, normalize(light.dir)));

    // lambertian diffuse: brightness proportional to angle btwn normal and light
    let diff    = max(dot(N, L), 0.0) * tex.rgb * light.color;
    // specular highlight: exponent 64 gives moderate-sized hotspot
    let spec    = pow(max(dot(N, H), 0.0), 64.0) * light.color * 0.4;
    // ambient: small constant so shadowed surfaces aren't completely black
    let ambient = 0.06 * tex.rgb;

    return vec4<f32>(ambient + spot * (diff + spec) * light.intensity, 1.0);
}
"#;

// State
// bind groups:
// cam_bg — every frame (camera orbits + eye_pos changes)
// model_bg — every frame (cube spin)
// tex_bg — static
// light_bg — only on L keypress (light switch)

struct State
{
    window:     Arc<Window>,
    gpu:        GpuCtx,
    depth_view: wgpu::TextureView,
    pipeline:   wgpu::RenderPipeline,

    vbuf:        wgpu::Buffer,
    ibuf:        wgpu::Buffer,
    index_count: u32,

    cam_buf:   wgpu::Buffer,  cam_bg:   wgpu::BindGroup,
    model_buf: wgpu::Buffer,  model_bg: wgpu::BindGroup,
    tex_bg:    wgpu::BindGroup,
    light_buf: wgpu::Buffer,  light_bg: wgpu::BindGroup,

    light_idx:  usize,
    ortho:      bool,
    start_time: Instant,
}

impl State
{
    async fn new(window: Arc<Window>) -> Self
    {
        let gpu = GpuCtx::new(window.clone()).await;

        // depth texture
        let depth_view = make_depth_texture(&gpu.device, gpu.config.width, gpu.config.height);

        // camera BGL needs VERTEX | FRAGMENT because vs_main uses view_proj
        // and fs_main uses eye_pos to compute specular half-vector
        let cam_bgl   = uniform_bgl(&gpu.device, "Camera BGL",
                                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT);
        let model_bgl = uniform_bgl(&gpu.device, "Model BGL",
                                    wgpu::ShaderStages::VERTEX);
        let light_bgl = uniform_bgl(&gpu.device, "Light BGL",
                                    wgpu::ShaderStages::FRAGMENT);

        // texture bind group layout has two bindings: texture_2d + sampler
        // not a uniform buffer so can't use uniform_bgl helper
        let tex_bgl = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Texture BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        // Float { filterable: true } = sampler may use bilinear filtering
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // filtering sampler - must match texture's filterable:true
                    ty:    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let id_cam    = CameraUniform {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            eye_pos:   [0.0; 3],
            _pad:      0.0,
        };
        let cam_buf   = uniform_buf(&gpu.device, bytemuck::bytes_of(&id_cam));
        let cam_bg    = uniform_bg(&gpu.device, &cam_bgl, &cam_buf);

        let id_model  = Model { model: Mat4::IDENTITY.to_cols_array_2d() };
        let model_buf = uniform_buf(&gpu.device, bytemuck::bytes_of(&id_model));
        let model_bg  = uniform_bg(&gpu.device, &model_bgl, &model_buf);

        // light init with preset 0, updated only on key press
        let light_u   = build_light_uniform(0);
        let light_buf = uniform_buf(&gpu.device, bytemuck::bytes_of(&light_u));
        let light_bg  = uniform_bg(&gpu.device, &light_bgl, &light_buf);

        // checkerboard texture
        // create_texture_with_data uploads pixel buffer directly instead of write_texture calls
        let checker     = make_checkerboard();
        let checker_tex = upload_texture(
            &gpu.device, &gpu.queue,
            &checker, 64, 64,
            wgpu::TextureFormat::Rgba8UnormSrgb,  // sRGB decode on sample
        );
        let tex_view = checker_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // sampler: Repeat wrapping for checkerboard tiling
        // linear filtering for anisotropic filtering
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // wire texture view + sampler into the texture bind group
        let tex_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Texture BG"),
            layout:  &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // compile shader and build render pipeline
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        // pipeline layout slots must match @group(N) order in the WGSL shader
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&cam_bgl), Some(&model_bgl), Some(&tex_bgl), Some(&light_bgl)],
            immediate_size:     0,
        });

        let pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     gpu.format(),
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                // cull_mode: Some(Face::Back) //switch back to this after checking winding order
                cull_mode: None,
                ..Default::default()
            },
            // depth/stencil state
            // depth_write_enabled: Some(true) = write depth of every passed frag
            // depth_compare: Less = discard frags behind what's already drawn
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // cube vert and index buffers
        let (cube_v, cube_i) = make_cube();
        let vbuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("VBuf"),
            contents: bytemuck::cast_slice(&cube_v),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let ibuf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("IBuf"),
            contents: bytemuck::cast_slice(&cube_i),
            usage:    wgpu::BufferUsages::INDEX,
        });
        let index_count = cube_i.len() as u32;

        Self {
            window, gpu, depth_view, pipeline,
            vbuf, ibuf, index_count,
            cam_buf, cam_bg,
            model_buf, model_bg,
            tex_bg,
            light_buf, light_bg,
            light_idx: 0,
            ortho:     false,
            start_time: Instant::now(),
        }
    }

    // update GPU config + depth texture if needed
    fn resize(&mut self, w: u32, h: u32)
    {
        self.gpu.resize(w, h);
        if w > 0 && h > 0 {
            self.depth_view = make_depth_texture(&self.gpu.device, w, h);
        }
    }

    // advance to next spotlight preset via LightUniform
    fn cycle_light(&mut self)
    {
        self.light_idx = (self.light_idx + 1) % PRESET_COUNT;
        let u = build_light_uniform(self.light_idx);
        self.gpu.queue.write_buffer(&self.light_buf, 0, bytemuck::bytes_of(&u));
    }

    fn toggle_ortho(&mut self) { self.ortho = !self.ortho; }

    fn render(&mut self) -> Result<(), ()>
    {
        self.window.request_redraw();
        if !self.gpu.is_configured { return Ok(()); }
        let t = self.start_time.elapsed().as_secs_f32();

        // camera orbit
        let cam = build_camera_uniform(t * 0.5, self.gpu.aspect(), self.ortho);
        self.gpu.queue.write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&cam));

        // cube spin
        let spin  = glam::Mat4::from_rotation_y(t * 0.35)
                  * glam::Mat4::from_rotation_x(0.3);
        let model = Model { model: spin.to_cols_array_2d() };
        self.gpu.queue.write_buffer(&self.model_buf, 0, bytemuck::bytes_of(&model));

        // acquire swap-chain texture
        let output = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(o)
            | wgpu::CurrentSurfaceTexture::Suboptimal(o) => o,
            wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost => {
                self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
                return Ok(());
            }
            // transient error - skip frame
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
            // validation failure - propagate as error to exit event loop
            wgpu::CurrentSurfaceTexture::Validation => return Err(()),
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Encoder") }
        );

        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    depth_slice:    None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // clear to bg
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.04, b: 0.08, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // clear depth, standard alg from class
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes:    None,
                multiview_mask:      None,
            });

            rpass.set_pipeline(&self.pipeline);
            // bind groups must be set in slot order matching @group(N) in shader
            rpass.set_bind_group(0, &self.cam_bg,   &[]);
            rpass.set_bind_group(1, &self.model_bg, &[]);
            rpass.set_bind_group(2, &self.tex_bg,   &[]);
            rpass.set_bind_group(3, &self.light_bg, &[]);
            rpass.set_vertex_buffer(0, self.vbuf.slice(..));
            rpass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        self.gpu.queue.submit(std::iter::once(enc.finish()));
        output.present();
        Ok(())
    }
}

// App (OS event handling)
// native: State lives directly in App as Option<State>
// wasm: State lives in Rc<RefCell<Option<State>>> 
// with_state() abstracts the platform difference behind a closure, this is why rust is the goat

pub struct App
{
    #[cfg(not(target_arch = "wasm32"))]
    state: Option<State>,
    #[cfg(target_arch = "wasm32")]
    state: std::rc::Rc<std::cell::RefCell<Option<State>>>,
}

impl App
{
    pub fn new() -> Self
    {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            state: None,
            #[cfg(target_arch = "wasm32")]
            state: std::rc::Rc::new(std::cell::RefCell::new(None)),
        }
    }

    fn with_state<R>(&mut self, f: impl FnOnce(&mut State) -> R) -> Option<R>
    {
        #[cfg(not(target_arch = "wasm32"))]
        { self.state.as_mut().map(f) }
        #[cfg(target_arch = "wasm32")]
        { self.state.borrow_mut().as_mut().map(f) }
    }
}

impl ApplicationHandler for App
{
    // called when event loop is ready, and on mobile/wasm suspend/resume
    // early-return guard prevents double-init
    fn resumed(&mut self, event_loop: &ActiveEventLoop)
    {
        if self.with_state(|_| ()).is_some() { return; }

        let window = Arc::new(event_loop.create_window(
            Window::default_attributes()
                .with_title("CMSC427 Lab 03 – Transforms, Textures & Lights")
                .with_inner_size(winit::dpi::PhysicalSize::new(1024u32, 720u32))
        ).unwrap());

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("canvas-host")
                               .or_else(|| d.body().map(|b| b.into())))
                .and_then(|host| window.canvas().and_then(|c| host.append_child(&c).ok()));
        }

        // pollster::block_on() spins the thread until the future resolves
        // safe here bc we're not inside an async runtime
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(State::new(window)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let cell   = self.state.clone();
            let window = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = State::new(window.clone()).await;
                *cell.borrow_mut() = Some(state);
                window.request_redraw();
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id:        winit::window::WindowId,
        event:      WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(s) => {
                self.with_state(|state| state.resize(s.width, s.height));
            }

            WindowEvent::RedrawRequested => {
                self.with_state(|state| { let _ = state.render(); });
            }

            // only match physical key-down events, ignore key-up and text input
            WindowEvent::KeyboardInput {
                event: winit::event::KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state:        ElementState::Pressed,
                    ..
                },
                ..
            } => match key {
                KeyCode::Escape => event_loop.exit(),
                KeyCode::KeyO   => { self.with_state(|s| s.toggle_ortho()); }
                KeyCode::KeyL   => { self.with_state(|s| s.cycle_light()); }
                _ => {}
            },

            _ => {}
        }
    }
}

// Entry points

pub fn run()
{
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn start()
{
    console_log::init_with_level(log::Level::Warn).ok();
    console_error_panic_hook::set_once();

    use winit::platform::web::EventLoopExtWebSys;
    let event_loop = EventLoop::new().unwrap();
    event_loop.spawn_app(App::new());
}
