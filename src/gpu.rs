use bytemuck::{Pod, Zeroable};
use std::sync::mpsc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub eye: [f32; 3],
    pub _pad0: f32,
    pub inv_vp: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub _pad1: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SimParams {
    pub g: f32,
    pub bh_radius: f32,
    pub c_sim: f32,
    pub disk_inner: f32,
    pub disk_outer: f32,
    // Pad to place light_pos at offset 48 (WGSL std140-like rules)
    // 5*f32 (20 bytes) + 7*f32 (28 bytes) = 48 bytes
    pub _pad_before_light: [f32; 7],
    // vec3 + trailing pad f32 (shader packs this as vec3 + pad)
    pub light_pos: [f32; 3],
    pub _pad0: f32,
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_layout: wgpu::BindGroupLayout,
    pub camera_buf: wgpu::Buffer,
    pub params_buf: wgpu::Buffer,
}

impl GpuContext {
    pub async fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let shader_src = include_str!("shader.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bh-compute"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute-bind-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bh-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params-uniform"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self { device, queue, pipeline, bind_layout, camera_buf, params_buf })
    }

    pub fn update_camera(&self, cam: &CameraUniform) {
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(cam));
    }

    pub fn update_params(&self, params: &SimParams) {
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }
    
    pub fn create_output_buffer(&self, width: u32, height: u32) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output-buffer"),
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn dispatch(&self, out_buf: &wgpu::Buffer, width: u32, height: u32) -> wgpu::Buffer {
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute-bind"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            ],
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (width * height * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind, &[]);
            let wg_x = (width + 7) / 8;
            let wg_y = (height + 7) / 8;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(out_buf, 0, &staging, 0, (width * height * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        staging
    }

    pub fn read_buffer_blocking(&self, staging: &wgpu::Buffer, size: usize) -> Vec<u8> {
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut out = vec![0u8; size];
        out.copy_from_slice(&data);
        drop(data);
        staging.unmap();
        out
    }
}
