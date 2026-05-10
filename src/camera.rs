
// camera.rs — camera uniform + orbit logic

// rendering 3d requires:
// View matrix: moves world so camera is at origin looking down -Z
// Mat4::look_at_rh(eye, target, up): orthonormal camera frame + world→camera transform

// Projection matrix: maps camera frustum to NDC cube, 2 diff types:
// Perspective — divides by depth (w), far objects appear smaller realistic 3d look, controlled by FOV + near/far clip
// Orthographic — no depth division, parallel lines stay parallel, objects same size regardless of depth

// combined `view_proj = proj * view` uploaded as mat4x4<f32> uniform
// vert shader: clip = view_proj * model * position



use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

// CameraUniform
// matches camera struct in lib.rs
// eye_pos needed by frag shader for specular highlights:
// Blinn-Phong half-vector H = normalize(L + V) as per class

// Note: GPU alignment rules:
// fields need to be 16 byte aligned, holdover from c
// view_proj: [[f32;4];4] — 64 bytes (already 16-byte aligned)
// eye_pos:[f32;3] — 12 bytes
// _pad:f32 —  4 bytes ← to reach 16 byte alignment for the *next* field

// bytemuck's Pod check caught mismatch at compile time instead of me being confused for like an hour
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform
{
    pub view_proj: [[f32; 4]; 4],
    pub eye_pos:   [f32; 3], 
    pub _pad:      f32,
}

// eye traces a horizontal circle of radius 3.5 at height y=2.0
// always pointing back toward origin
// _rh = right-handed coordinates (WebGPU NDC is right-handed, depth [0,1])
pub fn build_camera_uniform(angle: f32, aspect: f32, ortho: bool) -> CameraUniform
{
    // orbit position: circle in XZ plane at height y=2
    let eye  = Vec3::new(3.5 * angle.cos(), 2.0, 3.5 * angle.sin());

    // look_at_rh(eye, target, up):
    //  builds orthonormal camera frame where -Z points from eye toward target
    //  ret: the world→camera (view) matrix
    let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);

    let proj = if ortho {
        // orthographic box
        let h = 2.0;
        Mat4::orthographic_rh(-h * aspect, h * aspect, -h, h, 0.1, 100.0)
    } else {
        // perspective: 45 degree vertical FOV
        // near plane at 0.1 (avoid z-fighting), far at 100
        // _rh maps clip-space depth to [0,1] (WebGPU/D3D convention)
        Mat4::perspective_rh(45_f32.to_radians(), aspect, 0.1, 100.0)
    };

    CameraUniform {
        // proj * view transforms: object space → world → camera → clip
        view_proj: (proj * view).to_cols_array_2d(),
        eye_pos:   eye.to_array(),
        _pad:      0.0,
    }
}
