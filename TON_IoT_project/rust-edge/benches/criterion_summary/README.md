# Criterion Benchmark Summary

This directory contains the raw `estimates.json` files extracted from
`target/criterion/` (which is excluded from Git by `.gitignore` because it
also holds compiled Rust artefacts).  Copying only the small JSON files here
lets reviewers verify the latency figures in the paper without needing to
re-compile the codebase.

**How to regenerate:** from `TON_IoT_project/rust-edge/`, run
```
cargo bench
```
Criterion writes new `estimates.json` files under `target/criterion/`.

---

## Paper claims and their source locations

| Paper location | Claim | Source file | Criterion statistic |
|---|---|---|---|
| Table B1, §5.5 | **76.50 ns** per single-device frame update | `aatc_engine_single_device/process_frame/estimates.json` | `slope.point_estimate` |
| Table B1, §5.5 | **79.34 µs** for 1 000-device batch | `aatc_engine_multi_device/tracked_devices/1000/estimates.json` | `slope.point_estimate` |
| Table B1 | **8.60 µs** for 100-device batch | `aatc_engine_multi_device/tracked_devices/100/estimates.json` | `slope.point_estimate` |
| §5.5 | **16.08 ns** push-and-evict (sliding window) | `sliding_window_aggregation/push_and_evict/estimates.json` | `slope.point_estimate` |
| §5.5 | **456.7 ns** aggregate (16 × 44 → 88) | `sliding_window_aggregation/aggregate_16x44_to_88/estimates.json` | `slope.point_estimate` |

Criterion reports all durations in **nanoseconds** unless noted.  The
`slope` statistic is the linear-regression estimate over the full
measurement range and is the value reported in the paper (it is the most
stable estimate when sample counts are large).

---

## Full estimates (verbatim from Criterion output)

### `aatc_engine_single_device/process_frame`

```json
{
  "slope": {
    "confidence_interval": {
      "confidence_level": 0.95,
      "lower_bound": 74.648002834573200,
      "upper_bound": 78.619702090532600
    },
    "point_estimate": 76.496656501326440,
    "standard_error":  1.020218896680975
  },
  "mean": {
    "confidence_interval": {
      "confidence_level": 0.95,
      "lower_bound": 73.958245111869690,
      "upper_bound": 75.962863217179330
    },
    "point_estimate": 74.866745301518530,
    "standard_error":  0.514292110812036
  },
  "median": {
    "confidence_interval": {
      "confidence_level": 0.95,
      "lower_bound": 73.281073927226000,
      "upper_bound": 74.224726103049140
    },
    "point_estimate": 73.722392335418020,
    "standard_error":  0.243004614553380
  }
}
```

**Paper value: 76.50 ns** (rounded from slope = 76.4967 ns).
95 % CI: 74.65 – 78.62 ns.

---

### `aatc_engine_multi_device/tracked_devices/1`

```json
{
  "slope": {
    "point_estimate": 87.376533290589650,
    "confidence_interval": { "lower_bound": 86.703056169046100, "upper_bound": 88.239439451989180 }
  }
}
```

87.38 ns (1 device).

---

### `aatc_engine_multi_device/tracked_devices/10`

```json
{
  "slope": {
    "point_estimate": 823.924420935226000,
    "confidence_interval": { "lower_bound": 813.170131965545900, "upper_bound": 837.188052786289600 }
  }
}
```

823.9 ns (10 devices).

---

### `aatc_engine_multi_device/tracked_devices/100`

```json
{
  "slope": {
    "point_estimate": 8604.030032683118,
    "confidence_interval": { "lower_bound": 8500.896209066022, "upper_bound": 8722.622969828497 }
  }
}
```

**8.60 µs** (100 devices). 95 % CI: 8.50 – 8.72 µs.

---

### `aatc_engine_multi_device/tracked_devices/500`

```json
{
  "slope": {
    "point_estimate": 41288.730103812620,
    "confidence_interval": { "lower_bound": 41135.320417916670, "upper_bound": 41469.085461178640 }
  }
}
```

41.29 µs (500 devices).

---

### `aatc_engine_multi_device/tracked_devices/1000`

```json
{
  "slope": {
    "confidence_interval": {
      "confidence_level": 0.95,
      "lower_bound": 78992.241209149970,
      "upper_bound": 79685.193839095290
    },
    "point_estimate": 79337.702854122380,
    "standard_error":   176.685037490367020
  },
  "mean": {
    "confidence_interval": {
      "confidence_level": 0.95,
      "lower_bound": 77582.691521640260,
      "upper_bound": 78511.581317052350
    },
    "point_estimate": 78049.022409127650,
    "standard_error":   237.006518589450760
  }
}
```

**Paper value: 79.34 µs** (rounded from slope = 79337.7 ns = 79.338 µs).
95 % CI: 78.99 – 79.69 µs.

---

### `sliding_window_aggregation/push_and_evict`

```json
{
  "slope": {
    "point_estimate": 16.078901402235413,
    "confidence_interval": { "lower_bound": 15.932024267159477, "upper_bound": 16.272803959497598 }
  }
}
```

**16.08 ns** per push-and-evict operation.

---

### `sliding_window_aggregation/aggregate_16x44_to_88`

```json
{
  "slope": {
    "point_estimate": 456.683144461265670,
    "confidence_interval": { "lower_bound": 453.398177473409650, "upper_bound": 459.846072302669770 }
  }
}
```

**456.7 ns** per aggregation pass.
