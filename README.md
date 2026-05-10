# CMSC427 Lab 03 Report: Transforms, Textures & Lights

This lab renders a checkerboard-textured cube lit by a configurable spotlight, viewed through an orbiting camera that can toggle between perspective and orthographic projection. It extends lab 02 with surface normals (needed for lighting), texture sampling, a depth buffer (needed for correct face ordering), and a Blinn-Phong shading model. The main structural addition is four bind groups — one per data stream — which is the maximum wgpu supports by default. 

Again, I used the same [learn-wgpu tutorial](https://sotrh.github.io/learn-wgpu/) reference throughout and AI tools for understanding concepts only. VSCode extensions were used for linting.

## Vertex Normals

Each vertex now carries a surface normal in addition to position and UV coordinates, bringing the per-vertex size from 24 bytes (lab 02) to 32 bytes:

```rust
struct Vertex {
    position: [f32; 3],  // object-space XYZ
    normal:   [f32; 3],  // outward surface normal
    uv:       [f32; 2],  // texture coordinate
}
```

A cube has 8 unique corner positions, but each corner belongs to three faces that point in different directions. If adjacent faces shared a corner vertex, the GPU would interpolate their normals across the edge, producing smooth shading on what should be a hard corner. To avoid this, each face gets its own four vertices with the face's flat normal baked in: 6 faces × 4 vertices = 24 vertices total.

In the vertex shader, the normal is transformed by the Model matrix using `w=0.0` instead of `w=1.0`. The `w=1.0` used for positions means "apply translations"; `w=0.0` means "directions only, skip translations". Without this, translating the cube would incorrectly shift all its normals.

## Blinn-Phong Lighting

The fragment shader computes a three-term Blinn-Phong sum:

```wgsl
let diff    = max(dot(N, L), 0.0) * tex.rgb * light.color;
let spec    = pow(max(dot(N, H), 0.0), 64.0) * light.color * 0.4;
let ambient = 0.06 * tex.rgb;
return vec4<f32>(ambient + spot * (diff + spec) * light.intensity, 1.0);
```

The diffuse term uses the Lambertian model: brightness is proportional to the cosine of the angle between the surface normal N and the light direction L. The specular term uses the Blinn half-vector H = normalize(L + V) where V is the direction to the camera eye. The exponent 64 gives a moderately-sized specular highlight. The ambient constant (0.06) ensures surfaces in shadow are not completely black.

## Spotlight

The light is a spotlight rather than a point light. It has an inner cone (20° half-angle) that receives full intensity, and an outer cone (35°) where the light fades to zero:

```wgsl
let spot = smoothstep(light.cos_outer, light.cos_inner, dot(-L, normalize(light.dir)));
```

The cone boundaries are stored as cosines rather than angles to avoid an `acos()` in the shader — the dot product already gives cosines, so comparing directly is cheaper. `smoothstep` produces a smooth cubic fade between the outer and inner cones rather than a hard cutoff. Three presets selectable with the L key position the spotlight from different angles with different colour temperatures.

## Texture Mapping

The checkerboard texture is generated on the CPU at startup: a 64×64 RGBA buffer of 8×8 amber and indigo tiles. It's uploaded once with `create_texture_with_data()` and never changed. The format is `Rgba8UnormSrgb`, which means the GPU automatically converts from the stored sRGB values to linear light during sampling — important for correct lighting math.

The sampler uses bilinear filtering (`FilterMode::Linear`) and `AddressMode::Repeat`, so the checkerboard tiles across all six cube faces. The UV range [0,1] is mapped across each face independently in `make_cube()`.

## Depth Testing

Without a depth buffer, the GPU draws triangles in submission order. A back face submitted after a front face would overwrite it, making the cube look inside-out. The depth attachment solves this:

- `Depth32Float` texture, one f32 per pixel, stores the depth of the closest fragment seen so far
- `CompareFunction::Less` discards any fragment whose depth is ≥ the stored value
- Cleared to 1.0 (far plane) at the start of each frame so every first fragment passes

The depth texture must match the swap chain's pixel dimensions exactly, so it is recreated in `resize()` whenever the window changes size.

## Bind Group Architecture

This lab uses all four of wgpu's default bind group slots. Each slot holds a different data stream:

| Slot | Content | Updated |
|------|---------|---------|
| 0 | Camera (view_proj + eye_pos) | Every frame |
| 1 | Model matrix | Every frame |
| 2 | Texture + sampler | Never (static) |
| 3 | Light (spotlight params) | On L keypress |

The Model matrix is kept separate from the camera's view_proj for the same reason as in lab 02: updating it means writing 64 bytes to a buffer rather than re-uploading 768 bytes of vertex data. The eye position is included in the camera uniform because the fragment shader needs it to compute the specular half-vector.

## wgpu Porting Notes

In WebGL, depths are managed via framebuffer. In wgpu, the depth texture is created and managed explicitly, and the render pass needs a `depth_stencil_attachment` with its own `load`/`store` ops and clear value.

Texture sampling in wgpu is also done by separate sampler objects that are bound independently alongside the texture in the bind group. The bind group layout must declare both a `Texture` entry and a `Sampler` entry.

## Result

The final output is a spinning cube with a checkerboard texture lit by a configurable spotlight. The O key toggles between perspective and orthographic projection, making the difference between the two immediately visible. The L key cycles through three spotlight presets with different positions and colour temperatures. The depth buffer correctly handles all face orderings as the cube rotates.
