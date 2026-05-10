
// mesh.rs — geometry, textures, and depth buffer, refactored for lab3
// added surface normals + uv texturing for lighting and texturing in vertices
// added depth texture for correct face occlusion

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex
{
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
}

impl Vertex
{
    // same C layout screwery for GPU
    // offsets must exactly match Rust struct layout or GPU reads garbage
    pub fn layout() -> wgpu::VertexBufferLayout<'static>
    {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset:          0,
                    shader_location: 0,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // normal
                wgpu::VertexAttribute {
                    offset:          std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format:          wgpu::VertexFormat::Float32x3,
                },
                // uv
                wgpu::VertexAttribute {
                    offset:          std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format:          wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

// Cube geometry
// returns (vertices, indices) ready to upload to VERTEX and INDEX buffers
// face winding: ccw from outside
// cull_mode: None = draw both sides

pub fn make_cube() -> (Vec<Vertex>, Vec<u16>)
{
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idx:   Vec<u16>    = Vec::new();

    // shared uv corners
    let uvs = [[0.0f32, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    // face closure: appends 4 verts + 6 indices for one quad
    let mut face = |positions: [[f32; 3]; 4], normal: [f32; 3]| {
        let base = verts.len() as u16;
        for i in 0..4 {
            verts.push(Vertex { position: positions[i], normal, uv: uvs[i] });
        }
        idx.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    };

    // six faces, each with outward normal
    // +Z face is "front" in right-handed space (camera looks down -Z)
    face([[-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5], [ 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]], [ 0.0, 0.0, 1.0]);  // +Z front
    face([[ 0.5,-0.5,-0.5], [-0.5,-0.5,-0.5], [-0.5, 0.5,-0.5], [ 0.5, 0.5,-0.5]], [ 0.0, 0.0,-1.0]);  // -Z back
    face([[ 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5], [ 0.5, 0.5,-0.5], [ 0.5, 0.5, 0.5]], [ 1.0, 0.0, 0.0]);  // +X right
    face([[-0.5,-0.5,-0.5], [-0.5,-0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, 0.5,-0.5]], [-1.0, 0.0, 0.0]);  // -X left
    face([[-0.5, 0.5, 0.5], [ 0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5], [-0.5, 0.5,-0.5]], [ 0.0, 1.0, 0.0]);  // +Y top
    face([[-0.5,-0.5,-0.5], [ 0.5,-0.5,-0.5], [ 0.5,-0.5, 0.5], [-0.5,-0.5, 0.5]], [ 0.0,-1.0, 0.0]);  // -Y bottom

    (verts, idx)
}

// Checkerboard texture
// generates a 64x64 RGBA pixel buffer with 8x8 tiles of amber and indigo
// div by 8 = 8 columns and 8 rows of tiles
// XOR/parity of tile col and row indices determines colour

pub fn make_checkerboard() -> Vec<u8>
{
    const W: usize = 64;
    let mut px = vec![0u8; W * W * 4];
    for y in 0..W {
        for x in 0..W {
            let (r, g, b) = if (x / 8 + y / 8) % 2 == 0 {
                (215u8, 140u8, 45u8) 
            } else {
                (35u8,  55u8,  130u8)  
            };
            let i = (y * W + x) * 4;
            px[i] = r; px[i+1] = g; px[i+2] = b; px[i+3] = 255;
        }
    }
    px
}

// ── Depth texture
// creates a fresh Depth32Float texture matching given dimensions
// returns a view suitable for use as depth attachment in a render pass
// depth test keeps frag only when its depth is *less* than stored value
// returning TextureView directly (not Texture) bc render pass attachment
// only needs the view. texture lives as long as its view via ref-counted handle

pub fn make_depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView
{
    device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("Depth"),
        size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,          // no mipmaps needed for depth buffer
        sample_count:    1,          // no MSAA
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Depth32Float,
        // RENDER_ATTACHMENT: can be used as depth/stencil attachment
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats:    &[],
    }).create_view(&wgpu::TextureViewDescriptor::default())
}

// static texture upload
// TEXTURE_BINDING = shader can sample this via textureSample()
// COPY_DST = upload path internally does a buffer-to-texture copy

pub fn upload_texture(
    device:  &wgpu::Device,
    queue:   &wgpu::Queue,
    pixels:  &[u8],
    w:       u32,
    h:       u32,
    format:  wgpu::TextureFormat,
) -> wgpu::Texture
{
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label:           Some("Texture"),
            size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        },
        // LayerMajor: for 2d textures w/ one layer = row-major
        wgpu::util::TextureDataOrder::LayerMajor,
        pixels,
    )
}
