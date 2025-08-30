use macroquad::prelude::*;
use rayon::prelude::*;

mod gpu;
use gpu::{GpuContext, CameraUniform as GpuCamera, SimParams as GpuParams};

// Simulation constants
const G: f64 = 6.9e-3;
const BLACK_HOLE_RADIUS: f32 = 15.0;
const C_SIM: f64 = 55.0;
const PHOTON_SPEED: f64 = 260.0;
const STEP_SIZE: f64 = 2.0;
const GRAVITY_BEND_SCALE: f64 = 1.8;
const DISK_BRIGHTNESS_BOOST: f64 = 1.4;
const AMBIENT_MIN: f64 = 0.22;

#[derive(Debug, Clone, Copy)]
struct Body { mass: f64, position: DVec3 }
struct Simulation { bodies: Vec<Body> }
#[derive(Clone, Copy)]
struct Star { pos: DVec3 }
struct Light { pos: DVec3 }

struct Camera { target: DVec3, yaw: f64, pitch: f64, radius: f64, fov_y: f32, up_hint: DVec3 }
impl Camera {
    fn position(&self) -> DVec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.target + DVec3::new(self.radius * cp * sy, self.radius * sp, -self.radius * cp * cy)
    }
    fn view_projection_matrix(&self) -> Mat4 {
        let (screen_w, screen_h) = (screen_width(), screen_height());
        let aspect = if screen_h > 0.0 { screen_w / screen_h } else { 1.0 };
        let eye = self.position();
        let view = Mat4::look_at_lh(eye.as_vec3(), self.target.as_vec3(), self.up_hint.as_vec3());
        let proj = Mat4::perspective_lh(self.fov_y.to_radians(), aspect, 0.1, 4000.0);
        proj * view
    }
}

fn clamp01(x: f32) -> f32 { x.max(0.0).min(1.0) }

#[macroquad::main("Black Hole Renderer")]
async fn main() {
    // Scene
    let black_hole = Body { mass: 9500.0, position: DVec3::ZERO };
    let simulation = Simulation { bodies: vec![black_hole] };
    let light = Light { pos: DVec3::new(-200.0, 100.0, -200.0) };

    // Stars
    let mut stars = Vec::with_capacity(2000);
    for _ in 0..2000 {
        stars.push(Star { pos: DVec3::new(
            macroquad::rand::gen_range(-1000.0, 1000.0),
            macroquad::rand::gen_range(-1000.0, 1000.0),
            macroquad::rand::gen_range(-1000.0, 1000.0),
        )});
    }

    // ASCII grid (CPU path)
    const ASCII_RAMP: [char; 10] = ['.', ':', '-', '=', '+', '*', '%', 'S', '#', '@'];
    const FONT_SIZE: f32 = 16.0;
    let (grid_w, grid_h);
    {
        let (sw, sh) = (screen_width(), screen_height());
        grid_w = (sw / FONT_SIZE * 2.0) as usize;
        grid_h = (sh / FONT_SIZE) as usize;
    }
    #[derive(Clone, Copy)]
    struct Cell { char: char, color: Color }
    let mut grid = vec![Cell { char: ' ', color: BLACK }; grid_w * grid_h];

    // GPU
    let gpu_ctx = match GpuContext::new().await { Ok(c) => Some(c), Err(_) => None };
    let mut gpu_output: Option<Vec<u8>> = None;
    let mut use_gpu = gpu_ctx.is_some();

    // Camera and input
    let mut camera = Camera { target: DVec3::ZERO, yaw: 0.0, pitch: 0.14, radius: 250.0, fov_y: 60.0, up_hint: DVec3::Y };
    let orbit_sensitivity = 0.005;
    let zoom_sensitivity = 20.0;
    let mut last_mouse = mouse_position();
    let mut user_override_timer = 0.0f64;
    let override_cooldown = 1.25f64;
    let auto_orbit_speed = 0.15f64;

    // Disk radii for CPU path
    let disk_inner_radius = BLACK_HOLE_RADIUS as f64 * 2.5;
    let disk_outer_radius = 120.0f64;

    loop {
        let (screen_w, screen_h) = (screen_width(), screen_height());
        let dt = get_frame_time() as f64;

        // Camera controls
        let (mx, my) = mouse_position();
        let dm = (mx - last_mouse.0, my - last_mouse.1);
        last_mouse = (mx, my);
        let mut user_input = false;
        if is_mouse_button_down(MouseButton::Left) {
            camera.yaw   -= dm.0 as f64 * orbit_sensitivity;
            camera.pitch -= dm.1 as f64 * orbit_sensitivity;
            camera.pitch = camera.pitch.clamp(-0.95, 0.95);
            user_input = true;
        }
        let wheel = mouse_wheel().1;
        if wheel.abs() > 0.0 { camera.radius = (camera.radius - wheel as f64 * zoom_sensitivity).clamp(30.0, 1500.0); user_input = true; }
        user_override_timer = if user_input { override_cooldown } else { (user_override_timer - dt).max(0.0) };
        if user_override_timer <= 0.0 { camera.yaw += auto_orbit_speed * dt; }

        // Toggle GPU
        if is_key_pressed(KeyCode::G) { use_gpu = !use_gpu; }

        let eye = camera.position();
        let view_proj_matrix = camera.view_projection_matrix();
        let inv_vp = view_proj_matrix.inverse();

        // Render
        if use_gpu {
            if let Some(ctx) = &gpu_ctx {
                let vp = view_proj_matrix.to_cols_array_2d();
                let inv = inv_vp.to_cols_array_2d();
                let cam_u = GpuCamera { eye: [eye.x as f32, eye.y as f32, eye.z as f32], _pad0: 0.0, vp, inv_vp: inv, screen_size: [screen_w, screen_h], font_size: FONT_SIZE, _pad1: 0.0 };
                ctx.update_camera(&cam_u);
                let params = GpuParams { g: G as f32, bh_radius: BLACK_HOLE_RADIUS, c_sim: C_SIM as f32, photon_speed: PHOTON_SPEED as f32, step_size: STEP_SIZE as f32, gravity_bend_scale: GRAVITY_BEND_SCALE as f32, disk_inner: disk_inner_radius as f32, disk_outer: disk_outer_radius as f32 };
                ctx.update_params(&params);
                let out = ctx.create_output_buffer(screen_w as u32, screen_h as u32);
                let staging = ctx.dispatch(&out, screen_w as u32, screen_h as u32);
                gpu_output = Some(ctx.read_buffer_blocking(&staging, screen_w as usize * screen_h as usize * 4));
            } else {
                use_gpu = false;
            }
        } else {
            let debug_idx = (grid_h / 2) * grid_w + (grid_w / 2);
            grid.par_iter_mut().enumerate().for_each(|(i, cell)| {
                let sx = (i % grid_w) as f32;
                let sy = (i / grid_w) as f32;
                *cell = Cell { char: ' ', color: BLACK };

                let cell_w = FONT_SIZE / 2.0; let cell_h = FONT_SIZE;
                let px = sx * cell_w + cell_w * 0.5; let py = sy * cell_h + cell_h * 0.5;
                let ndc_x = (px / screen_w) * 2.0 - 1.0; let ndc_y = 1.0 - (py / screen_h) * 2.0;
                let pw_far  = inv_vp.project_point3(vec3(ndc_x, ndc_y,  1.0));
                let ray_origin = eye; let ray_dir = (pw_far.as_dvec3() - ray_origin).normalize();

                if i == debug_idx { macroquad::logging::info!("ray origin=({:.2},{:.2},{:.2}) dir=({:.3},{:.3},{:.3})", ray_origin.x, ray_origin.y, ray_origin.z, ray_dir.x, ray_dir.y, ray_dir.z); }

                let mut ray_pos = ray_origin; let mut dir = ray_dir; let max_iter = 1200;
                for _ in 0..max_iter {
                    let to_bh = DVec3::ZERO - ray_pos; let dist_sq = to_bh.length_squared();
                    if dist_sq < (BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS) as f64 { break; }
                    if dist_sq > 0.0 { let r = dist_sq.sqrt(); let accel_mag = GRAVITY_BEND_SCALE * (G * simulation.bodies[0].mass) / dist_sq; let accel_vec = to_bh / r * accel_mag; let dt_step = STEP_SIZE / PHOTON_SPEED; dir = (dir + accel_vec * dt_step).normalize(); }
                    let y0 = ray_pos.y; let y1 = ray_pos.y + dir.y * STEP_SIZE;
                    if (y0 > 0.0 && y1 <= 0.0) || (y0 < 0.0 && y1 >= 0.0) {
                        let s = -y0 / dir.y; if s.is_finite() && s >= 0.0 && s <= STEP_SIZE {
                            let p = ray_pos + dir * s; let r = (p.x * p.x + p.z * p.z).sqrt();
                            if r > disk_inner_radius && r < disk_outer_radius {
                                let dir_to_light = (DVec3::new(light.pos.x, light.pos.y, light.pos.z) - p).normalize();
                                let oc = -p; let b = oc.dot(dir_to_light); let c = oc.dot(oc) - (BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS) as f64; let discriminant = b * b - c;
                                let mut brightness = if discriminant < 0.0 { (dir_to_light.dot(DVec3::Y)).max(AMBIENT_MIN) } else { AMBIENT_MIN };
                                let tangent = if r > 0.0 { DVec3::new(-p.z / r, 0.0, p.x / r) } else { DVec3::new(1.0, 0.0, 0.0) };
                                let v_k = (G * simulation.bodies[0].mass / r.max(1e-6)).sqrt(); let beta = (v_k / C_SIM).min(0.95); let gamma = 1.0 / (1.0 - beta * beta).sqrt();
                                let view_dir = (ray_origin - p).normalize(); let cos_theta = tangent.dot(view_dir).clamp(-1.0, 1.0);
                                let denom = gamma * (1.0 - beta * cos_theta); let doppler = if denom > 0.0 { 1.0 / denom } else { 0.0 }; let doppler_clamped = doppler.clamp(0.4, 2.5);
                                brightness *= doppler_clamped.powf(3.0); brightness *= DISK_BRIGHTNESS_BOOST; brightness = brightness.clamp(0.06, 4.0);
                                let ramp_idx = ((brightness * (ASCII_RAMP.len() - 1) as f64).round() as isize).clamp(0, (ASCII_RAMP.len() - 1) as isize) as usize;
                                let base = vec3(1.0, 0.6, 0.15); let blue_white = vec3(0.85, 0.90, 1.0); let red_dark = vec3(1.0, 0.25, 0.05);
                                let d = doppler_clamped as f32; let color_vec = if d >= 1.0 { base.lerp(blue_white, ((d - 1.0) / 1.5).min(1.0)) } else { base.lerp(red_dark, ((1.0 - d) / 0.6).min(1.0)) };
                                let scale = clamp01((brightness as f32) / 1.0); let col = Color::new(clamp01(color_vec.x * scale), clamp01(color_vec.y * scale), clamp01(color_vec.z * scale), 1.0);
                                *cell = Cell { char: ASCII_RAMP[ramp_idx], color: col }; break;
                            }
                        }
                    }
                    ray_pos += dir * STEP_SIZE; if ray_pos.length() > 5000.0 { break; }
                }
            });
        }

        // Draw
        clear_background(BLACK);
        for star in &stars {
            let sp = view_proj_matrix.project_point3(star.pos.as_vec3());
            let draw_x = (sp.x + 1.0) * 0.5 * screen_width();
            let draw_y = (1.0 - (sp.y + 1.0) * 0.5) * screen_height();
            if sp.z < 1.0 && draw_x > 0.0 && draw_x < screen_w && draw_y > 0.0 && draw_y < screen_h { draw_circle(draw_x, draw_y, 0.5, WHITE); }
        }

        if use_gpu {
            if let Some(bytes) = &gpu_output {
                let tex = Texture2D::from_rgba8(screen_w as u16, screen_h as u16, &bytes);
                draw_texture_ex(&tex, 0.0, 0.0, WHITE, DrawTextureParams { dest_size: Some(vec2(screen_w, screen_h)), ..Default::default()});
            }
        } else {
            for i in 0..grid.len() { let x = (i % grid_w) as f32 * FONT_SIZE / 2.0; let y = (i / grid_w) as f32 * FONT_SIZE; let cell = grid[i]; let mut buf = [0u8; 4]; let s = cell.char.encode_utf8(&mut buf); draw_text(s, x, y, FONT_SIZE, cell.color); }
        }

        if !use_gpu {
            let bh_screen = view_proj_matrix.project_point3(DVec3::ZERO.as_vec3());
            if bh_screen.z < 1.0 && bh_screen.z > -1.0 {
                let x = (bh_screen.x + 1.0) * 0.5 * screen_width(); let y = (1.0 - (bh_screen.y + 1.0) * 0.5) * screen_height();
                let forward = (camera.target - eye).normalize(); let right = DVec3::Y.cross(forward).normalize();
                let offset_world = (DVec3::ZERO + right * BLACK_HOLE_RADIUS as f64).as_vec3(); let offset_screen = view_proj_matrix.project_point3(offset_world);
                let ox = (offset_screen.x + 1.0) * 0.5 * screen_width(); let oy = (1.0 - (offset_screen.y + 1.0) * 0.5) * screen_height();
                let bh_px_radius = ((ox - x).powi(2) + (oy - y).powi(2)).sqrt(); draw_circle(x, y, bh_px_radius.max(2.0), BLACK);
            }
        }

        next_frame().await;
    }
}
