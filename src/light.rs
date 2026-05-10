
// light.rs — spotlight uniform and presets for lab03

// point light with a directional cone. 
// shines brightly inside narrow *inner cone* and fades to zero btwn the inner and outer cone angles
// soft-edge fade modelled with WGSL's smoothstep()

// cone boundary specified as *cosine* of half-angle as in class
// avoids an acos() in the shader - just compare dot products (which already
// give cosines) directly

// Presets/options (toggle with L):
// 0: upper-right + warm 
// 1: left cool 
// 2: overhead neutraL



use bytemuck::{Pod, Zeroable};
use glam::Vec3;



// Note: GPU Alignment
// cue 216 flashbacks
// struct layout:
//   pos:        [f32;3]  12 bytes  offset  0
//   _p0:        f32       4 bytes  offset 12  ← pad after pos
//   color:      [f32;3]  12 bytes  offset 16
//   intensity:  f32       4 bytes  offset 28  ← doubles as useful data
//   dir:        [f32;3]  12 bytes  offset 32
//   _p1:        f32       4 bytes  offset 44  ← pad after dir
//   cos_inner:  f32       4 bytes  offset 48
//   cos_outer:  f32       4 bytes  offset 52
//   _p2:        [f32;2]   8 bytes  offset 56  ← pad to 64 bytes total
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LightUniform
{
    pub pos:       [f32; 3],  pub _p0:       f32,   // world-space pon + pad
    pub color:     [f32; 3],  pub intensity: f32,   // RGB color * intensity
    pub dir:       [f32; 3],  pub _p1:       f32,   // spotlight direction + pad
    pub cos_inner: f32,                             // inner cone boundary
    pub cos_outer: f32,                             // outer cone boundary
    pub _p2:       [f32; 2],
}

// light positions, world space
const POSITIONS: [[f32; 3]; 3] = [
    [ 2.0, 3.0,  2.0],
    [-3.0, 1.0,  0.0],  
    [ 0.0, 5.0, -0.5],
];

// light colors
const COLORS: [[f32; 3]; 3] = [
    [1.0, 0.95, 0.85],  
    [0.8, 0.88, 1.0 ],  
    [1.0, 1.0,  1.0 ], 
];

// builds the uniform given idx
pub fn build_light_uniform(idx: usize) -> LightUniform
{
    let pos = Vec3::from(POSITIONS[idx]);
    LightUniform {
        pos:       pos.to_array(),
        _p0:       0.0,
        color:     COLORS[idx],
        intensity: 1.8,  // multiplied with diffuse + specular in shader
        // direction: from light toward origin
        dir:       (-pos.normalize()).to_array(),
        _p1:       0.0,
        cos_inner: 20_f32.to_radians().cos(),
        cos_outer: 35_f32.to_radians().cos(),
        _p2:       [0.0; 2],
    }
}

pub const PRESET_COUNT: usize = POSITIONS.len();
