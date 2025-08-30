struct CameraUniform {
  eye: vec3<f32>,
  _pad0: f32,
  vp: mat4x4<f32>,
  inv_vp: mat4x4<f32>,
  screen_size: vec2<f32>,
  font_size: f32,
  _pad1: f32,
};

struct SimParams {
  g: f32,
  bh_radius: f32,
  c_sim: f32,
  photon_speed: f32,
  step_size: f32,
  gravity_bend_scale: f32,
  disk_inner: f32,
  disk_outer: f32,
};

@group(0) @binding(0) var<uniform> cam: CameraUniform;
@group(0) @binding(1) var<uniform> sim: SimParams;
@group(0) @binding(2) var<storage, read_write> out_buf: array<u32>;

fn clamp01(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(cam.screen_size.x);
  let height = u32(cam.screen_size.y);
  if (gid.x >= width || gid.y >= height) { return; }

  let px = f32(gid.x) + 0.5;
  let py = f32(gid.y) + 0.5;
  let ndc_x = (px / cam.screen_size.x) * 2.0 - 1.0;
  let ndc_y = 1.0 - (py / cam.screen_size.y) * 2.0;
  let p_near = cam.inv_vp * vec4<f32>(ndc_x, ndc_y, -1.0, 1.0);
  let p_far  = cam.inv_vp * vec4<f32>(ndc_x, ndc_y,  1.0, 1.0);
  let wp_near = p_near.xyz / p_near.w;
  let wp_far  = p_far.xyz / p_far.w;
  let eye = cam.eye;
  var dir = normalize((wp_far - eye));
  var pos = eye;

  let max_iter = 1200u;
  var color = vec3<f32>(0.0);

  for (var i: u32 = 0u; i < max_iter; i = i + 1u) {
    let to_bh = -pos;
    let dist_sq = max(1e-6, dot(to_bh, to_bh));
    if (dist_sq < sim.bh_radius * sim.bh_radius) {
      color = vec3<f32>(0.0);
      break;
    }

    let r = sqrt(dist_sq);
    let accel_mag = sim.gravity_bend_scale * (sim.g * 9500.0) / dist_sq;
    let accel = (to_bh / r) * accel_mag;
    let dt = sim.step_size / sim.photon_speed;
    dir = normalize(dir + accel * dt);

    let y0 = pos.y;
    let y1 = pos.y + dir.y * sim.step_size;
    if ((y0 > 0.0 && y1 <= 0.0) || (y0 < 0.0 && y1 >= 0.0)) {
      let s = -y0 / dir.y;
      if (s >= 0.0 && s <= sim.step_size) {
        let p = pos + dir * s;
        let r_hit = length(vec2<f32>(p.x, p.z));
        if (r_hit > sim.disk_inner && r_hit < sim.disk_outer) {
          // Simple shade: brighter towards center
          let b = clamp(1.5 - r_hit / sim.disk_outer, 0.0, 1.0);
          color = vec3<f32>(b, b * 0.7, 0.4 * b);
          break;
        }
      }
    }

    pos = pos + dir * sim.step_size;
    if (length(pos) > 5000.0) {
      break;
    }
  }

  let rgba = vec4<f32>(color, 1.0);
  let u = vec4<u32>(rgba * 255.0);
  let idx = gid.y * width + gid.x;
  out_buf[idx] = (u.x & 0xFFu) | ((u.y & 0xFFu) << 8u) | ((u.z & 0xFFu) << 16u) | (0xFFu << 24u);
}
