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
    bh_mass: f32,
    _pad_a: vec2<f32>,
    light_pos: vec4<f32>,
    planet_pos_mass: vec4<f32>,
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

    var accum = vec3<f32>(0.0);
    var alpha = 0.0;

    // Tilted disk plane (constant across ray)
    let tilt = radians(3.0);
    let n = normalize(vec3<f32>(sin(tilt), cos(tilt), 0.0));
    let half_h = 10.0; // half thickness of disk volume

    // 2. Ray Marching Loop (reduced steps + adaptive step length)
    for (var i: u32 = 0u; i < 400u; i = i + 1u) { // Reduced from 600
        let to_bh = -pos;
        let dist_sq = max(1e-6, dot(to_bh, to_bh));
        if (dist_sq < sim.bh_radius * sim.bh_radius) {
            accum = vec3<f32>(0.0); // Hit the event horizon, color is black
            alpha = 1.0; // Make it opaque
            break; 
        }

        // Bend the ray's path with gravity (BH + planet) using inverse sqrt where possible
        let rsq_bh = dist_sq;
        let inv_r_bh = inverseSqrt(rsq_bh);
        let accel_mag_bh = 1.8 * (sim.g * sim.bh_mass) / rsq_bh;
        let accel_bh = to_bh * inv_r_bh * accel_mag_bh;
        let to_pl = sim.planet_pos_mass.xyz - pos;
        let rsq_pl = max(1e-6, dot(to_pl, to_pl));
        let inv_r_pl = inverseSqrt(rsq_pl);
        let accel_mag_pl = 1.8 * (sim.g * sim.planet_pos_mass.w) / rsq_pl;
        let accel_pl = to_pl * inv_r_pl * accel_mag_pl;
        let accel = accel_bh + accel_pl;
        let dt = 2.0 / 200.0;
        dir = normalize(dir + accel * dt);

        // Volumetric sampling around tilted disk plane within a finite thickness
        let d_plane = dot(n, pos);
        if (abs(d_plane) < half_h) {
            let radial = pos - n * d_plane;
            let r_hit = length(radial.xz);
            let edge_softness = 40.0;
            let inner_soft = smoothstep(sim.disk_inner - edge_softness, sim.disk_inner + edge_softness, r_hit);
            let outer_soft = 1.0 - smoothstep(sim.disk_outer - edge_softness, sim.disk_outer + edge_softness, r_hit);
            let disk_mask = clamp(inner_soft * outer_soft, 0.0, 1.0);
            if (disk_mask > 0.001) {
                let h_t = 1.0 - clamp(abs(d_plane) / half_h, 0.0, 1.0);
                let radial_dir = normalize(vec3<f32>(radial.x, radial.y, radial.z));
                let tangent = normalize(cross(n, radial_dir));
                let v_k = sqrt((sim.g * sim.bh_mass) / max(1e-5, r_hit));
                let beta = min(v_k / sim.c_sim, 0.95);
                let gamma = 1.0 / sqrt(1.0 - beta * beta);
                let view_dir = normalize(cam.eye - pos);
                let cos_theta = dot(tangent, view_dir);
                let doppler = 1.0 / (gamma * (1.0 - beta * cos_theta));
                var brightness = 0.98 * disk_mask * h_t; // slightly boosted baseline for visibility
                brightness = brightness * clamp(doppler, 0.4, 2.5);
                // Slightly warmer yellow bias near the BH
                let col_yellow = vec3(1.0, 0.98, 0.38);
                let col_orange = vec3(1.0, 0.55, 0.12);
                let col_red    = vec3(1.0, 0.16, 0.08);
                let t_dop = clamp((doppler - 0.6) / 1.2, 0.0, 1.0);
                // Bias more toward yellow (about +12%)
                let warm = mix(col_orange, col_yellow, clamp(t_dop + 0.12, 0.0, 1.0));
                let t_rad = clamp((r_hit - sim.disk_inner) / max(1e-5, (sim.disk_outer - sim.disk_inner)), 0.0, 1.0);
                let color_vec = mix(warm, col_red, sqrt(t_rad)); // cheaper than pow
                // Higher per-step alpha for a denser, more opaque disk
                let a = min(0.6, 0.12 * brightness);
                let one_m_a = 1.0 - alpha;
                accum += color_vec * a * one_m_a;
                alpha += a * one_m_a;
            }
        }
        if (alpha > 0.995) { break; }
        // Adaptive step: smaller near disk and near BH, larger farther away
        let step_len = clamp(sqrt(dist_sq) / 100.0, 1.0, 5.0);
        pos = pos + dir * step_len;
        if (length(pos) > 5000.0) { break; }
    }

    // 4. Write final color to the output buffer
    let rgba = vec4<f32>(accum, alpha);
    let u = vec4<u32>(rgba * 255.0);
    let idx = gid.y * width + gid.x;
    out_buf[idx] = (u.x & 0xFFu) | ((u.y & 0xFFu) << 8u) | ((u.z & 0xFFu) << 16u) | (u32(alpha * 0.95 * 255.0) << 24u);
}
