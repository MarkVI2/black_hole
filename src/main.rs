use macroquad::prelude::*;
use rayon::prelude::*;

mod gpu;
use gpu::{GpuContext, CameraUniform as GpuCamera, SimParams as GpuParams};

// --- TUNING PARAMETERS ---
const G: f64 = 6.674e-3;
const BLACK_HOLE_RADIUS: f32 = 15.0;
const C_SIM: f64 = 100.0;

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
    let black_hole = Body { mass: 10000.0, position: DVec3::ZERO };
    let light = Light { pos: DVec3::new(-200.0, 100.0, -200.0) };

    // Add a second body (planet/sun) for lensing
    let planet = Body { mass: 5000.0, position: DVec3::new(120.0, 0.0, 0.0) };
    let simulation = Simulation { bodies: vec![black_hole, planet] };

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

    // --- Grid setup ---
    const GRID_SIZE: usize = 301; // 151 points for 150 segments
    let mut grid_points = vec![DVec3::ZERO; GRID_SIZE * GRID_SIZE];
    let grid_spacing = 150.0 / 150.0;
    for j in 0..GRID_SIZE {
        for i in 0..GRID_SIZE {
            let x = (i as f64 - 150.0) * grid_spacing;
            let z = (j as f64 - 150.0) * grid_spacing;
            grid_points[j * GRID_SIZE + i] = DVec3::new(x, -20.0, z);
        }
    }

    loop {
        let dt = get_frame_time() as f64;
        let (screen_w, screen_h) = (screen_width(), screen_height());

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
            // Half-resolution for performance
            screen_size: [screen_w * 0.5, screen_h * 0.5],
                    _pad1: [0.0; 2],
                };
                // Wider, thicker disk radii for visibility
                let disk_inner_gpu = BLACK_HOLE_RADIUS * 1.6;
                let disk_outer_gpu = BLACK_HOLE_RADIUS * 4.5;
                let params = GpuParams {
                    g: G as f32,
                    bh_radius: BLACK_HOLE_RADIUS,
                    c_sim: C_SIM as f32,
                    disk_inner: disk_inner_gpu,
                    disk_outer: disk_outer_gpu,
                    bh_mass: black_hole.mass as f32,
                    _pad_a: [0.0; 2],
                    light_pos: [light.pos.x as f32, light.pos.y as f32, light.pos.z as f32, 0.0],
                    planet_pos_mass: [planet.position.x as f32, planet.position.y as f32, planet.position.z as f32, planet.mass as f32],
                };
                ctx.update_camera(&cam_u);
                ctx.update_params(&params);
                let w = (screen_w * 0.5) as u32;
                let h = (screen_h * 0.5) as u32;
                let out = ctx.create_output_buffer(w, h);
                let staging = ctx.dispatch(&out, w, h);
                let gpu_output = ctx.read_buffer_blocking(&staging, (w as usize) * (h as usize) * 4);
                let tex = Texture2D::from_rgba8(w as u16, h as u16, &gpu_output);
                gpu_tex = Some(tex);
            }
        } else {
            // ================== CPU PARALLEL CALCULATION ==================
            // Wider, thicker disk for CPU path
            let disk_inner_radius = BLACK_HOLE_RADIUS as f64 * 1.6;
            let disk_outer_radius = BLACK_HOLE_RADIUS as f64 * 4.5;
            let tilt_deg = 3.0f64; // 2.5 - 5 deg; pick 3 deg default
            let tilt = tilt_deg.to_radians();
            let n = DVec3::new(tilt.sin(), tilt.cos(), 0.0).normalize(); // tilt around X axis
            
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
                let mut accum = vec3(0.0f32, 0.0, 0.0);
                let mut alpha = 0.0f32;
                for _ in 0..600 { // Reduced from 1200
                    let to_bh = DVec3::ZERO - ray_pos;
                    let dist_sq = to_bh.length_squared();
                    if dist_sq < (BLACK_HOLE_RADIUS * BLACK_HOLE_RADIUS) as f64 { break; }
                    
                    // Adaptive step length
                    let step_len = (dist_sq.sqrt() / 100.0).clamp(1.0, 5.0);

                    // Multi-body gravity: BH + planet
                    let acc = if dist_sq > 0.0 {
                        let r = dist_sq.sqrt();
                        let accel_mag_bh = 1.8 * (G * simulation.bodies[0].mass) / dist_sq;
                        let accel_bh = to_bh / r * accel_mag_bh;
                        let to_pl = planet.position - ray_pos;
                        let d2_pl = to_pl.length_squared().max(1e-9);
                        let r_pl = d2_pl.sqrt();
                        let accel_mag_pl = 1.8 * (G * planet.mass) / d2_pl;
                        let accel_pl = to_pl / r_pl * accel_mag_pl;
                        accel_bh + accel_pl
                    } else { DVec3::ZERO };
                    dir = (dir + acc * (2.0 / 260.0)).normalize();
                    // Volumetric sampling near tilted plane within finite thickness
                    let d_plane = n.dot(ray_pos);
                    let half_h = 10.0f64; // half-thickness of disk volume
                    if d_plane.abs() < half_h {
                        let radial = ray_pos - n * d_plane;
                        let r = (radial.x * radial.x + radial.z * radial.z).sqrt();
                        let edge_softness = 40.0f64;
                        let inner_soft = ((r - (disk_inner_radius - edge_softness)) / (2.0 * edge_softness)).clamp(0.0, 1.0) as f32;
                        let outer_soft = 1.0 - ((r - (disk_outer_radius - edge_softness)) / (2.0 * edge_softness)).clamp(0.0, 1.0) as f32;
                        let disk_mask = (inner_soft * outer_soft).clamp(0.0, 1.0);
                        if disk_mask > 0.001 {
                            // Thickness falloff
                            let h_t = 1.0 - ((d_plane.abs() as f32) / half_h as f32).clamp(0.0, 1.0);
                            // Tangent direction in plane perpendicular to radial
                            let radial_dir = if r > 1e-6 { DVec3::new(radial.x / r, radial.y / r, radial.z / r) } else { DVec3::X };
                            let tangent = n.cross(radial_dir).normalize_or_zero();
                            let v_k = (G * simulation.bodies[0].mass / r.max(1e-6)).sqrt();
                            let beta = (v_k / C_SIM).min(0.95);
                            let gamma = 1.0 / (1.0 - beta * beta).sqrt();
                            let view_dir = (eye - ray_pos).normalize();
                            let cos_theta = tangent.dot(view_dir);
                            let doppler = 1.0 / (gamma * (1.0 - beta * cos_theta));
                            let mut brightness = 0.6f32 * disk_mask * h_t;
                            brightness *= doppler.clamp(0.4, 2.5) as f32;
                            let col_yellow = vec3(1.0, 0.95, 0.30);
                            let col_orange = vec3(1.0, 0.55, 0.12);
                            let col_red    = vec3(1.0, 0.16, 0.08);
                            let t_dop = (((doppler as f32) - 0.6) / 1.2).clamp(0.0, 1.0);
                            // Bias slightly more toward yellow (about +8%)
                            let warm = col_orange.lerp(col_yellow, (t_dop + 0.08).clamp(0.0, 1.0));
                            let t_rad = ((r - disk_inner_radius) / (disk_outer_radius - disk_inner_radius + 1e-6)).clamp(0.0, 1.0) as f32;
                            let color_vec = warm.lerp(col_red, t_rad.powf(1.2));
                            let a = (0.02f32 * brightness).min(0.25);
                            let one_m_a = 1.0 - alpha;
                            accum.x += color_vec.x * a * one_m_a;
                            accum.y += color_vec.y * a * one_m_a;
                            accum.z += color_vec.z * a * one_m_a;
                            alpha += a * one_m_a;
                        }
                    }
                    if alpha > 0.98 { break; }
                    ray_pos += dir * step_len;
                    if ray_pos.length() > 5000.0 { break; }
                }
                if alpha > 0.0 {
                    *pixel_color = Color::new(accum.x.min(1.0), accum.y.min(1.0), accum.z.min(1.0), 1.0);
                }
            });
        }
        
    // ================== SEQUENTIAL DRAWING ==================

    // Planet draws AFTER grid so it is not hidden by it

        // --- Draw BH silhouette before disk only in CPU mode (GPU shader handles BH) ---
        if !use_gpu {
            let bh_screen = view_proj_matrix.project_point3(DVec3::ZERO.as_vec3());
            if bh_screen.z < 1.0 {
                let x = (bh_screen.x + 1.0) * 0.5 * screen_w;
                let y = (1.0 - (bh_screen.y + 1.0) * 0.5) * screen_h;
                let forward = (DVec3::ZERO - eye).normalize();
                let right = forward.cross(camera.up_hint).normalize();
                let edge_world = right * BLACK_HOLE_RADIUS as f64;
                let edge_screen = view_proj_matrix.project_point3(edge_world.as_vec3());
                let ex = (edge_screen.x + 1.0) * 0.5 * screen_w;
                let ey = (1.0 - (edge_screen.y + 1.0) * 0.5) * screen_h;
                let screen_radius = ((ex - x).powi(2) + (ey - y).powi(2)).sqrt().abs();
                draw_circle(x, y, screen_radius.max(2.0), BLACK);
            }
        }

        // --- Draw the CPU Disk Colors (if CPU path) ---
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

        // --- Unified warped spacetime grid overlay (drawn for GPU and CPU paths) ---
        {
            let bh_mass = simulation.bodies[0].mass;
            let planet_mass = simulation.bodies[1].mass;
            let planet_pos = simulation.bodies[1].position;

        let mut warped_points_3d = vec![DVec3::ZERO; GRID_SIZE * GRID_SIZE];
        let mut strengths = vec![0.0f32; GRID_SIZE * GRID_SIZE];
            for j in 0..GRID_SIZE {
                for i in 0..GRID_SIZE {
                    let p_orig = grid_points[j * GRID_SIZE + i];
                    
                    // Simplified gravity warp for y-coord
                    let to_bh = -p_orig;
                    let d2_bh = to_bh.length_squared().max(1.0);
            let warp_y_bh = -100.0 * (G * bh_mass) / d2_bh.sqrt();

                    let to_pl = planet_pos - p_orig;
                    let d2_pl = to_pl.length_squared().max(1.0);
            let warp_y_pl = -100.0 * (G * planet_mass) / d2_pl.sqrt();
                    
                    let warped_y = warp_y_bh + warp_y_pl;

                    warped_points_3d[j * GRID_SIZE + i] = DVec3::new(p_orig.x, warped_y - 30.0, p_orig.z);

            // Per-point gravitational force magnitude (GM/r^2), summed from BH and planet
            let force_bh = (G * bh_mass) / d2_bh; // r^2 already
            let force_pl = (G * planet_mass) / d2_pl;
            let total_force = (force_bh + force_pl) as f32;
            // Nonlinear remap to emphasize small bends: t' = sqrt(clamp(k * t,0,1))
            let t_lin = (total_force * 4.5).clamp(0.0, 1.0) as f32; // slightly higher gain
            let t = t_lin.sqrt();
            strengths[j * GRID_SIZE + i] = t;
                }
            }

            let mut screen_points = vec![Vec2::ZERO; GRID_SIZE * GRID_SIZE];
            let mut points_in_frustum = vec![false; GRID_SIZE * GRID_SIZE];

            for i in 0..(GRID_SIZE*GRID_SIZE) {
                let p_3d = warped_points_3d[i];
                let p_screen = view_proj_matrix.project_point3(p_3d.as_vec3());
                if p_screen.z < 1.0 {
                    screen_points[i] = vec2((p_screen.x + 1.0) * 0.5 * screen_w, (1.0 - (p_screen.y + 1.0) * 0.5) * screen_h);
                    points_in_frustum[i] = true;
                }
            }

            // Cooler-to-deeper gradient; alpha kept same, but color span widened for contrast
            let light_grid = Color::new(0.72, 0.72, 1.00, 0.85);
            let deep_grid  = Color::new(0.05, 0.05, 0.45, 0.85);
            for j in 0..GRID_SIZE {
                for i in 0..GRID_SIZE {
                    if i < GRID_SIZE - 1 {
                        let idx1 = j * GRID_SIZE + i;
                        let idx2 = j * GRID_SIZE + i + 1;
                        if points_in_frustum[idx1] && points_in_frustum[idx2] {
                            let t = 0.5 * (strengths[idx1] + strengths[idx2]);
                            let col = Color::new(
                                light_grid.r + (deep_grid.r - light_grid.r) * t,
                                light_grid.g + (deep_grid.g - light_grid.g) * t,
                                light_grid.b + (deep_grid.b - light_grid.b) * t,
                                light_grid.a,
                            );
                            draw_line(screen_points[idx1].x, screen_points[idx1].y, screen_points[idx2].x, screen_points[idx2].y, 1.0, col);
                        }
                    }
                    if j < GRID_SIZE - 1 {
                        let idx1 = j * GRID_SIZE + i;
                        let idx2 = (j + 1) * GRID_SIZE + i;
                        if points_in_frustum[idx1] && points_in_frustum[idx2] {
                            let t = 0.5 * (strengths[idx1] + strengths[idx2]);
                            let col = Color::new(
                                light_grid.r + (deep_grid.r - light_grid.r) * t,
                                light_grid.g + (deep_grid.g - light_grid.g) * t,
                                light_grid.b + (deep_grid.b - light_grid.b) * t,
                                light_grid.a,
                            );
                            draw_line(screen_points[idx1].x, screen_points[idx1].y, screen_points[idx2].x, screen_points[idx2].y, 1.0, col);
                        }
                    }
                }
            }
        }

        // --- Draw the planet/sun for lensing visualization (after grid and disk so it appears above them) ---
        // Smoothly transition between planet visibility and an Einstein ring when it is behind the BH.
        let planet_screen = view_proj_matrix.project_point3(planet.position.as_vec3());
        // Screen-space BH position and size for alignment measure
        let bh_screen = view_proj_matrix.project_point3(DVec3::ZERO.as_vec3());
        let same_side = planet_screen.z < 1.0 && bh_screen.z < 1.0;
        let px = (planet_screen.x + 1.0) * 0.5 * screen_w;
        let py = (1.0 - (planet_screen.y + 1.0) * 0.5) * screen_h;
        let bx = (bh_screen.x + 1.0) * 0.5 * screen_w;
        let by = (1.0 - (bh_screen.y + 1.0) * 0.5) * screen_h;
        let screen_dist = ((px - bx).powi(2) + (py - by).powi(2)).sqrt();
        // BH's screen-space radius
        let forward = (DVec3::ZERO - eye).normalize();
        let right = forward.cross(camera.up_hint).normalize();
        let edge_world = right * BLACK_HOLE_RADIUS as f64;
        let edge_screen = view_proj_matrix.project_point3(edge_world.as_vec3());
        let ex = (edge_screen.x + 1.0) * 0.5 * screen_w;
        let ey = (1.0 - (edge_screen.y + 1.0) * 0.5) * screen_h;
        let bh_screen_radius = ((ex - bx).powi(2) + (ey - by).powi(2)).sqrt().abs().max(2.0);
        // Depth-based factor: positive when planet is behind the BH
        let camera_to_planet = planet.position - eye;
        let camera_to_bh = DVec3::ZERO - eye;
        let planet_dist = camera_to_planet.length();
        let bh_dist = camera_to_bh.length();
        let depth_occ = ((planet_dist - bh_dist) / 60.0).clamp(0.0, 1.0) as f32; // 60u soft range
        // Alignment factor: 1 when centered behind BH, 0 when far away
        let align = (1.0 - (screen_dist / (1.8 * bh_screen_radius)).clamp(0.0, 1.0)) as f32;
        // Combined smooth occlusion/lensing factor
        let occ = if same_side { (align.powf(1.2) * depth_occ).clamp(0.0, 1.0) } else { 0.0 };
        if planet_screen.z < 1.0 {
            // Static on-screen size regardless of zoom
            let planet_screen_radius = 15.0;
            // Fade planet as it goes behind/aligned; keep a tiny minimum so AA stays pleasant
            let planet_alpha = (1.0 - 0.95 * occ).clamp(0.05, 1.0);
            let sun_color = Color::new(1.0, 0.98, 0.35, planet_alpha);
            draw_circle(px, py, planet_screen_radius, sun_color);
            // Lensing halo (Einstein ring) with intensity based on occ
            if occ > 0.01 {
                let ring_r = bh_screen_radius * (1.12 + 0.10 * align as f32);
                let base_thickness = 6.0 * (0.35 + 0.65 * align as f32);
                let base_alpha = (0.65 * occ).clamp(0.0, 0.85);
                let base_col = Color::new(1.0, 0.98, 0.65, base_alpha);
                // Multi-pass to fake a soft glow with gentle falloff
                for k in 0..4 {
                    let kt = k as f32 / 3.0;
                    let r = ring_r + kt * 2.0;
                    let th = (base_thickness * (1.0 - kt * 0.35)).max(1.0);
                    let a = (base_col.a * (1.0 - kt * 0.4)).clamp(0.0, 1.0);
                    let col = Color::new(base_col.r, base_col.g, base_col.b, a);
                    draw_circle_lines(bx, by, r, th, col);
                }
            }
        }

        // --- Draw BH silhouette last in CPU mode to ensure it occludes the planet if overlapping ---
        if !use_gpu {
            let bh_screen = view_proj_matrix.project_point3(DVec3::ZERO.as_vec3());
            if bh_screen.z < 1.0 {
                let bx2 = (bh_screen.x + 1.0) * 0.5 * screen_w;
                let by2 = (1.0 - (bh_screen.y + 1.0) * 0.5) * screen_h;
                let forward2 = (DVec3::ZERO - eye).normalize();
                let right2 = forward2.cross(camera.up_hint).normalize();
                let edge_world2 = right2 * BLACK_HOLE_RADIUS as f64;
                let edge_screen2 = view_proj_matrix.project_point3(edge_world2.as_vec3());
                let ex2 = (edge_screen2.x + 1.0) * 0.5 * screen_w;
                let ey2 = (1.0 - (edge_screen2.y + 1.0) * 0.5) * screen_h;
                let screen_radius2 = ((ex2 - bx2).powi(2) + (ey2 - by2).powi(2)).sqrt().abs();
                draw_circle(bx2, by2, screen_radius2.max(2.0), BLACK);
            }
        }

        // Draw UI Text
        let ui_text = format!("Renderer: {} (Press G to toggle)", if use_gpu { "GPU" } else { "CPU" });
        draw_text(&ui_text, 10.0, 20.0, 20.0, WHITE);
        
        next_frame().await;
    }
}
