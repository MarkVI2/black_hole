// Structs to receive data from the CPU
// Match Rust #[repr(C)] layouts in gpu.rs exactly (padding included)
struct CameraUniform {
    eye: vec3<f32>,
    _pad0: f32,
    inv_vp: mat4x4<f32>,
    screen_size: vec2<f32>,
    _pad1: vec2<f32>,
};

struct SimParams {
    g: f32,
    bh_radius: f32,
    c_sim: f32,
    disk_inner: f32,
    disk_outer: f32,
    // align next vec3 on 16-byte boundary (matches Rust 7*f32 pad)
    _pad_before_light: vec3<f32>,
    light_pos: vec3<f32>,
    _pad0: f32,
};

// Bindings for the data buffers
@group(0) @binding(0) var<uniform> cam: CameraUniform;
@group(0) @binding(1) var<uniform> sim: SimParams;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

// The main compute shader function, executed for each pixel
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(cam.screen_size.x);
    let height = u32(cam.screen_size.y);
    if (gid.x >= width || gid.y >= height) { return; }

    // 1. Unproject pixel coordinate to get a 3D ray direction
    let px = f32(gid.x) + 0.5;
    let py = f32(gid.y) + 0.5;
    let ndc_x = (px / cam.screen_size.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (py / cam.screen_size.y) * 2.0;
    let p_far  = cam.inv_vp * vec4<f32>(ndc_x, ndc_y,  1.0, 1.0);
    let wp_far = p_far.xyz / p_far.w;
    
    var dir = normalize(wp_far - cam.eye);
    var pos = cam.eye;
    
    var color = vec3<f32>(0.0);
    var out_alpha: f32 = 0.0;

    // 2. Ray Marching Loop
    for (var i: u32 = 0u; i < 1200u; i = i + 1u) {
        let to_bh = -pos;
        let dist_sq = max(1e-6, dot(to_bh, to_bh));
        if (dist_sq < sim.bh_radius * sim.bh_radius) {
            color = vec3<f32>(0.0);
            break;
        }

        // Bend the ray's path with gravity
        let r = sqrt(dist_sq);
        let accel_mag = 1.8 * (sim.g * 9500.0) / dist_sq;
        let accel = (to_bh / r) * accel_mag;
        let dt = 2.0 / 260.0;
        dir = normalize(dir + accel * dt);

        // Check for intersection with the accretion disk plane
        let y0 = pos.y;
        let y1 = pos.y + dir.y * 2.0;
        if ((y0 > 0.0 && y1 <= 0.0) || (y0 < 0.0 && y1 >= 0.0)) {
            let s = -y0 / dir.y;
            if (s >= 0.0 && s <= 2.0) {
                let p = pos + dir * s;
                let r_hit = length(vec2<f32>(p.x, p.z));

                // Make disk much thicker and add smoothing for a cloud effect
                // Increase disk thickness by expanding the range and smoothing edges
                let disk_thickness = 55.0;
                let disk_fade = 30.0;
                let disk_center = (sim.disk_inner + sim.disk_outer) * 0.5;
                let disk_radius = (sim.disk_outer - sim.disk_inner) * 0.5;
                let dist_to_center = abs(r_hit - disk_center);
                let fade = smoothstep(disk_radius - disk_thickness, disk_radius + disk_thickness + disk_fade, dist_to_center);
                if (r_hit > sim.disk_inner - disk_thickness && r_hit < sim.disk_outer + disk_thickness) {
                    // 3. Shading and Doppler Effect (no shadow)
                    var brightness = 0.35;
                    let tangent = normalize(vec3<f32>(-p.z, 0.0, p.x));
                    let v_k = sqrt((sim.g * 9500.0) / r_hit);
                    let beta = min(v_k / sim.c_sim, 0.95);
                    let gamma = 1.0 / sqrt(1.0 - beta * beta);
                    let view_dir = normalize(cam.eye - p);
                    let cos_theta = dot(tangent, view_dir);
                    let doppler = 1.0 / (gamma * (1.0 - beta * cos_theta));
                    brightness = brightness * pow(clamp(doppler, 0.4, 2.5), 3.0) * 1.4;

                    // Smooth gradient between deep orange and yellow
                    let deep_orange = vec3(1.0, 0.62, 0.12);
                    let bright_yellow = vec3(1.0, 0.96, 0.55);
                    let t = clamp((doppler - 0.7) / 1.2, 0.0, 1.0);
                    let color_vec = mix(deep_orange, bright_yellow, smoothstep(0.0, 1.0, t));

                    // Smooth fade, no pulse
                    let smooth_alpha = 1.0 - fade;
                    color = color_vec * clamp(brightness * smooth_alpha, 0.0, 1.0);
                    out_alpha = 1.0;
                    break;
                }
            }
        }
        pos = pos + dir * 2.0;
        if (length(pos) > 5000.0) { break; }
    }

    // 4. Projected 3D grid in XZ plane (y=0), mostly at bottom of screen
    let grid_spacing = 20.0;
    let grid_thickness = 0.02;
    var grid_alpha = 0.0;
    let grid_color = vec3<f32>(0.5, 0.5, 0.7);
    // Use the initial view ray (no bending) to intersect plane y=0
    let ray_dir0 = normalize(wp_far - cam.eye);
    let denom = ray_dir0.y;
    if (abs(denom) > 1e-5) {
        let t_plane = -cam.eye.y / denom;
        if (t_plane > 0.0) {
            let hit = cam.eye + ray_dir0 * t_plane;
            // Only draw near the bottom of the screen
            let ndcY = ndc_y; // from earlier
            let bottom_factor = clamp((0.0 - ndcY) / 1.0, 0.0, 1.0);
            let gx = abs(fract(hit.x / grid_spacing) - 0.5);
            let gz = abs(fract(hit.z / grid_spacing) - 0.5);
            if (gx < grid_thickness || gz < grid_thickness) {
                grid_alpha = 0.7 * bottom_factor * (1.0 - out_alpha);
                color = mix(color, grid_color, grid_alpha);
                if (grid_alpha > out_alpha) {
                    out_alpha = grid_alpha;
                }
            }
        }
    }

    // 5. Write final color to the output buffer.
    let rgba = vec4<f32>(color, out_alpha);
    let u = vec4<u32>(rgba * 255.0);
    let idx = gid.y * width + gid.x;
    out_buf[idx] = (u.x & 0xFFu) | ((u.y & 0xFFu) << 8u) | ((u.z & 0xFFu) << 16u) | ((u.w & 0xFFu) << 24u);
}
