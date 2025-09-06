Black Hole Renderer
===================

This project is a real-time 2D/3D simulation of a black hole, demonstrating gravitational lensing and an accretion disk. It is built using Rust and features both a CPU and a GPU-accelerated rendering backend.

Features
--------
- **Gravitational Lensing**: Simulates the bending of light paths around massive objects. This is visualized by ray tracing from the camera and deflecting the rays based on the gravity of the black hole and a nearby planet.
- **Accretion Disk**: A glowing disk of matter is rendered around the black hole, with Doppler shifting and relativistic beaming effects that make the side rotating towards the camera appear brighter and bluer, while the side rotating away appears dimmer and redder.
- **Warped Spacetime Grid**: A 2D grid is drawn, and its vertical position is warped by the gravitational fields of the black hole and the planet, providing a visual representation of curved spacetime.
- **Dual Rendering Backends**:
    - **CPU Renderer**: A parallelized ray tracer using `rayon` that calculates the final color of each pixel.
    - **GPU Renderer**: A compute shader-based renderer using `wgpu` for high performance.
- **Interactive Camera**: The camera can be orbited around the scene and zoomed in and out.
- **Einstein Ring**: When the planet moves behind the black hole from the camera's perspective, an Einstein ring (a halo of lensed light from the planet) becomes visible.

Controls
--------
- **Toggle Renderer**: Press the 'G' key to switch between the CPU and GPU rendering backends.
- **Orbit Camera**: Click and drag the left mouse button to rotate the camera around the black hole.
- **Zoom**: Use the mouse scroll wheel to zoom in and out.

Technical Details
-----------------
- **Language**: Rust
- **Graphics & Windowing**: `macroquad`
- **GPU Computing**: `wgpu`
- **CPU Parallelism**: `rayon`
- **Dependencies**: See `Cargo.toml` for a full list.

How to Run
----------
1. Make sure you have Rust and Cargo installed.
2. Clone the repository.
3. Run the project with `cargo run --release`. The `--release` flag is highly recommended for performance.
