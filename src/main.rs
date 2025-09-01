use macroquad::prelude::*;
use rayon::prelude::*;

mod gpu;
use gpu::{GpuContext, CameraUniform as GpuCamera, SimParams as GpuParams};

// --- TUNING PARAMETERS ---
const G: f64 = 6.674e-2;
const BLACK_HOLE_RADIUS: f32 = 15.0;
const C_SIM: f64 = 55.0;

#[derive(Debug, Clone, Copy)]
struct Body { mass: f64, position: DVec3 }
struct Simulation { bodies: Vec<Body> }
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

#[macroquad::main("Black Hole Pixel Renderer")]
async fn main() {
    // Scene Setup
    let black_hole = Body { mass: 9500.0, position: DVec3::ZERO };
    let light = Light { pos: DVec3::new(-200.0, 100.0, -200.0) };

    // Add a second body (planet/sun) for lensing
    let planet = Body { mass: 3000.0, position: DVec3::new(120.0, 0.0, 0.0) };
    let simulation = Simulation { bodies: vec![black_hole, planet] };
    let planet_color = YELLOW;

    // --- GPU Context and Toggle ---
    let gpu_ctx = pollster::block_on(GpuContext::new()).ok();
    let mut use_gpu = gpu_ctx.is_some();

    // --- CPU Pixel Grid Setup ---
    let step_size: usize = 3;
    let (screen_w, screen_h) = (screen_width(), screen_height());
    let (grid_w, grid_h) = ((screen_w / step_size as f32) as usize, (screen_h / step_size as f32) as usize);
    let mut cpu_grid = vec![BLACK; grid_w * grid_h];
    
    // --- Camera and Input Setup ---
    let mut camera = Camera {
        target: DVec3::ZERO,
        yaw: 0.0,
        pitch: 0.05,
        radius: 250.0,
        fov_y: 60.0,
        up_hint: DVec3::Y,
    };
    let mut last_mouse = mouse_position();
    let mut user_override_timer = 0.0f64;

    loop {
        let dt = get_frame_time() as f64;
        let (screen_w, screen_h) = (screen_width(), screen_height());

    // Always clear at the start of the frame before any drawing
    clear_background(BLACK);

        // --- Camera Controls ---
        let (mx, my) = mouse_position();
        let dm = (mx - last_mouse.0, my - last_mouse.1);
        last_mouse = (mx, my);
        let mut user_input = false;
        if is_mouse_button_down(MouseButton::Left) {
            camera.yaw -= dm.0 as f64 * 0.005;
            camera.pitch -= dm.1 as f64 * 0.005;
            camera.pitch = camera.pitch.clamp(-1.5, 1.5);
            user_input = true;
        }
        let wheel = mouse_wheel().1;
        if wheel.abs() > 0.0 {
            camera.radius = (camera.radius - wheel as f64 * 20.0).clamp(40.0, 1500.0);
            user_input = true;
        }
        user_override_timer = if user_input { 1.25 } else { (user_override_timer - dt).max(0.0) };
        if user_override_timer <= 0.0 { camera.yaw += 0.15 * dt; }
        
        // --- GPU/CPU Toggle ---
        if is_key_pressed(KeyCode::G) { use_gpu = !use_gpu && gpu_ctx.is_some(); }

        let eye = camera.position();
        let view_proj_matrix = camera.view_projection_matrix();
        let inv_vp = view_proj_matrix.inverse();

        let mut gpu_tex: Option<Texture2D> = None;
        if use_gpu {
            if let Some(ctx) = &gpu_ctx {
                // ================== GPU RENDER PATH (compute + prepare texture) ==================
                let cam_u = GpuCamera {
                    eye: eye.as_vec3().to_array(),
                    _pad0: 0.0,
                    inv_vp: inv_vp.to_cols_array_2d(),
                    screen_size: [screen_w, screen_h],
                    _pad1: [0.0; 2],
                };
                let params = GpuParams {
                    g: G as f32,
                    bh_radius: BLACK_HOLE_RADIUS,
                    c_sim: C_SIM as f32,
                    disk_inner: BLACK_HOLE_RADIUS as f32 * 2.5,
                    disk_outer: 120.0,
                    _pad_before_light: [0.0; 7],
                    light_pos: light.pos.as_vec3().to_array(),
                    _pad0: 0.0,
                };
                ctx.update_camera(&cam_u);
                ctx.update_params(&params);
                let out = ctx.create_output_buffer(screen_w as u32, screen_h as u32);
                let staging = ctx.dispatch(&out, screen_w as u32, screen_h as u32);
                let gpu_output = ctx.read_buffer_blocking(&staging, screen_w as usize * screen_h as usize * 4);
                
                let tex = Texture2D::from_rgba8(screen_w as u16, screen_h as u16, &gpu_output);
                gpu_tex = Some(tex);
            }
        } else {
            // ================== CPU PARALLEL CALCULATION ==================
            let disk_inner_radius = BLACK_HOLE_RADIUS as f64 * 2.5;
            let disk_outer_radius = 120.0f64;
            
            cpu_grid.par_iter_mut().enumerate().for_each(|(i, pixel_color)| {
                let sx = (i % grid_w) as f32 * step_size as f32;
                let sy = (i / grid_w) as f32 * step_size as f32;
                *pixel_color = BLACK;

                let ndc_x = (sx / screen_w) * 2.0 - 1.0;
                let ndc_y = 1.0 - (sy / screen_h) * 2.0;
                let pw_far = inv_vp.project_point3(vec3(ndc_x, ndc_y, 1.0));
                let ray_dir = (pw_far.as_dvec3() - eye).normalize();

                let mut ray_pos = eye;
                let mut dir = ray_dir;
                for _ in 0..1200 {
                    let to_bh = DVec3::ZERO - ray_pos;
                    let dist_sq = to_bh.length_squared();
                    if dist_sq < (BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS) as f64 { break; }
                    if dist_sq > 0.0 {
                        let r = dist_sq.sqrt();
                        let accel_mag = 1.8 * (G * simulation.bodies[0].mass) / dist_sq;
                        let accel_vec = to_bh / r * accel_mag;
                        dir = (dir + accel_vec * (2.0 / 260.0)).normalize();
                    }
                    let y0 = ray_pos.y;
                    let y1 = ray_pos.y + dir.y * 2.0;
                    if (y0 > 0.0 && y1 <= 0.0) || (y0 < 0.0 && y1 >= 0.0) {
                        let s = -y0 / dir.y;
                        if s.is_finite() && s >= 0.0 && s <= 2.0 {
                            let p = ray_pos + dir * s;
                            let r = (p.x * p.x + p.z * p.z).sqrt();
                            if r > disk_inner_radius && r < disk_outer_radius {
                                let dir_to_light = (light.pos - p).normalize();
                                let oc = -p;
                                let b = oc.dot(dir_to_light);
                                let c = oc.dot(oc) - (BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS) as f64;
                                let mut brightness = if b * b - c < 0.0 { (dir_to_light.dot(DVec3::Y)).max(0.22) } else { 0.22 };
                                let tangent = if r > 0.0 { DVec3::new(-p.z / r, 0.0, p.x / r) } else { DVec3::X };
                                let v_k = (G * simulation.bodies[0].mass / r.max(1e-6)).sqrt();
                                let beta = (v_k / C_SIM).min(0.95);
                                let gamma = 1.0 / (1.0 - beta * beta).sqrt();
                                let view_dir = (eye - p).normalize();
                                let cos_theta = tangent.dot(view_dir);
                                let doppler = 1.0 / (gamma * (1.0 - beta * cos_theta));
                                brightness *= doppler.clamp(0.4, 2.5).powf(3.0) * 1.4;
                                
                                let base = vec3(0.1, 0.5, 1.0);
                                let blue_white = vec3(0.8, 0.9, 1.0);
                                let red_dark = vec3(0.7, 0.1, 1.0);
                                let d = doppler.clamp(0.4, 2.5) as f32;
                                let color_vec = if d >= 1.0 { base.lerp(blue_white, ((d - 1.0) / 1.5).min(1.0)) } else { base.lerp(red_dark, ((1.0 - d) / 0.6).min(1.0)) };
                                let b = brightness as f32;
                                *pixel_color = Color::new(color_vec.x * b, color_vec.y * b, color_vec.z * b, 1.0);
                                break;
                            }
                        }
                    }
                    ray_pos += dir * 2.0;
                    if ray_pos.length() > 5000.0 { break; }
                }
            });
        }
        
    // ================== SEQUENTIAL DRAWING ==================

        // --- Draw the planet/sun for lensing visualization ---
        // Hide planet if occluded by black hole
        let planet_screen = view_proj_matrix.project_point3(planet.position.as_vec3());
        let camera_to_planet = planet.position - eye;
        let camera_to_bh = DVec3::ZERO - eye;
        let planet_dist = camera_to_planet.length();
        let bh_dist = camera_to_bh.length();
        let camera_to_planet_dir = camera_to_planet.normalize();
        let camera_to_bh_dir = camera_to_bh.normalize();
        let dot_dirs = camera_to_planet_dir.dot(camera_to_bh_dir);
        let occluded = dot_dirs > 0.999 && bh_dist < planet_dist && bh_dist < BLACK_HOLE_RADIUS as f64 * 1.2;
        if planet_screen.z < 1.0 && !occluded {
            let px = (planet_screen.x + 1.0) * 0.5 * screen_w;
            let py = (1.0 - (planet_screen.y + 1.0) * 0.5) * screen_h;
            draw_circle(px, py, 32.0, planet_color); // Increased radius
        }

        // --- Draw the CPU Pixel Grid on top, but skip black (transparent) cells ---
        if !use_gpu {
            for i in 0..cpu_grid.len() {
                let c = cpu_grid[i];
                if c.r == 0.0 && c.g == 0.0 && c.b == 0.0 { continue; }
                let x = (i % grid_w) as f32 * step_size as f32;
                let y = (i / grid_w) as f32 * step_size as f32;
                draw_rectangle(x, y, step_size as f32, step_size as f32, c);
            }
        }
        // --- Composite GPU texture with alpha over the stars ---
        if use_gpu {
            if let Some(tex) = gpu_tex.take() {
                draw_texture_ex(&tex, 0.0, 0.0, WHITE, DrawTextureParams { dest_size: Some(vec2(screen_w, screen_h)), ..Default::default()});
            }
        }

    // --- Draw the Black Hole Silhouette on top ---
        let bh_screen = view_proj_matrix.project_point3(DVec3::ZERO.as_vec3());
        if bh_screen.z < 1.0 {
            let x = (bh_screen.x + 1.0) * 0.5 * screen_w;
            let y = (1.0 - (bh_screen.y + 1.0) * 0.5) * screen_h;
            let right_edge_world = DVec3::X * BLACK_HOLE_RADIUS as f64;
            let right_edge_screen = view_proj_matrix.project_point3((eye + (right_edge_world - eye).normalize() * 10.0).as_vec3());
            let ox = (right_edge_screen.x + 1.0) * 0.5 * screen_w;
            let oy = (1.0 - (right_edge_screen.y + 1.0) * 0.5) * screen_h;
            let screen_radius = ((ox - x).powi(2) + (oy - y).powi(2)).sqrt();
            draw_circle(x, y, screen_radius.max(2.0), BLACK);
        }

        // Draw UI Text
        let ui_text = format!("Renderer: {} (Press G to toggle)", if use_gpu { "GPU" } else { "CPU" });
        draw_text(&ui_text, 10.0, 20.0, 20.0, WHITE);
        
        next_frame().await;
    }
}
