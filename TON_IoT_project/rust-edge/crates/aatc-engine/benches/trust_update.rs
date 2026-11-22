use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use aatc_engine::{AatcEngine, EngineConfig, TelemetryFrame};
use std::collections::VecDeque;

const FEATURE_DIM: usize = 44;
const WINDOW_SIZE: usize = 16;
const MODEL_INPUT_DIM: usize = FEATURE_DIM * 2;

struct BenchDeviceWindow {
    frames: VecDeque<[f32; FEATURE_DIM]>,
}

impl BenchDeviceWindow {
    fn new() -> Self {
        Self {
            frames: VecDeque::with_capacity(WINDOW_SIZE + 1),
        }
    }

    fn push(&mut self, frame: [f32; FEATURE_DIM]) {
        self.frames.push_back(frame);
        if self.frames.len() > WINDOW_SIZE {
            self.frames.pop_front();
        }
    }

    fn aggregate(&self) -> [f32; MODEL_INPUT_DIM] {
        let n = self.frames.len() as f32;
        let mut means = [0f32; FEATURE_DIM];
        let mut stds = [0f32; FEATURE_DIM];

        for frame in &self.frames {
            for (i, &v) in frame.iter().enumerate() {
                means[i] += v / n;
            }
        }

        if self.frames.len() > 1 {
            let mut variance = [0f32; FEATURE_DIM];
            for frame in &self.frames {
                for (i, &v) in frame.iter().enumerate() {
                    let diff = v - means[i];
                    variance[i] += diff * diff / n;
                }
            }
            for (s, &var) in stds.iter_mut().zip(variance.iter()) {
                *s = var.sqrt();
            }
        }

        let mut result = [0f32; MODEL_INPUT_DIM];
        result[..FEATURE_DIM].copy_from_slice(&means);
        result[FEATURE_DIM..].copy_from_slice(&stds);
        result
    }
}

fn bench_trust_update_single_device(c: &mut Criterion) {
    let mut group = c.benchmark_group("aatc_engine_single_device");
    let mut engine = AatcEngine::new(EngineConfig::default());
    let dev_id = "device_sensor_01".to_string();

    // Pre-warm state
    for t in 0..50u64 {
        engine.process(&TelemetryFrame {
            device_id: dev_id.clone(),
            anomaly_score: 0.05,
            timestamp_ms: t * 100,
        });
    }

    let mut tick = 50u64;
    group.bench_function("process_frame", |b| {
        b.iter(|| {
            tick += 1;
            let score = if tick % 10 == 0 { 0.65 } else { 0.02 };
            black_box(engine.process(&TelemetryFrame {
                device_id: dev_id.clone(),
                anomaly_score: score,
                timestamp_ms: tick * 100,
            }))
        });
    });

    group.finish();
}

fn bench_trust_update_multi_device(c: &mut Criterion) {
    let mut group = c.benchmark_group("aatc_engine_multi_device");

    for &n_devices in &[1, 10, 100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("tracked_devices", n_devices),
            &n_devices,
            |b, &n| {
                let mut engine = AatcEngine::new(EngineConfig::default());
                let device_ids: Vec<String> = (0..n).map(|d| format!("192.168.1.{}", d % 254)).collect();

                // Pre-warm: build state for all devices
                for (i, dev_id) in device_ids.iter().enumerate() {
                    for t in 0..10u64 {
                        engine.process(&TelemetryFrame {
                            device_id: dev_id.clone(),
                            anomaly_score: 0.01,
                            timestamp_ms: (t * 100) + (i as u64),
                        });
                    }
                }

                let mut tick = 10u64;
                b.iter(|| {
                    tick += 1;
                    for dev_id in &device_ids {
                        let score = if tick % 7 == 0 { 0.72 } else { 0.03 };
                        black_box(engine.process(&TelemetryFrame {
                            device_id: dev_id.clone(),
                            anomaly_score: score,
                            timestamp_ms: tick * 100,
                        }));
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_sliding_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("sliding_window_aggregation");
    let mut window = BenchDeviceWindow::new();
    let frame = [0.42f32; FEATURE_DIM];

    // Pre-fill window to full capacity (16 frames)
    for _ in 0..WINDOW_SIZE {
        window.push(frame);
    }

    group.bench_function("push_and_evict", |b| {
        b.iter(|| {
            window.push(black_box(frame));
        });
    });

    group.bench_function("aggregate_16x44_to_88", |b| {
        b.iter(|| {
            black_box(window.aggregate())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_trust_update_single_device,
    bench_trust_update_multi_device,
    bench_sliding_window
);
criterion_main!(benches);
