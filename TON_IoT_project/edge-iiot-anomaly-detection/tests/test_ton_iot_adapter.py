"""Unit tests verifying the ToN-IoT data adapter, preprocessor, and calibration protocol."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from config.settings import config
from src.data.ton_iot_adapter import (
    TON_IOT_FEATURE_COLUMNS,
    TON_IOT_REQUIRED_COLUMNS,
    TonIoTDataAdapter,
    TonIoTChronologicalSplitter,
)
from src.preprocessing.ton_iot_preprocessor import TonIoTPreprocessor
from src.model.ton_iot_calibration import (
    TonIoTScoreNormalizer,
    compute_benign_anomaly_threshold,
    compute_benign_gamma,
    simulate_ema_stream,
)


@pytest.fixture
def synthetic_modbus_df():
    """Create synthetic Modbus telemetry across Train, Calib, and Eval dates."""
    dates = [
        # Train interval: 31-Mar to 03-Apr 2019
        "31-Mar-19", "01-Apr-19", "02-Apr-19", "03-Apr-19",
        # Calibration interval: 04-Apr 2019
        "04-Apr-19", "04-Apr-19",
        # Evaluation interval: 23-Apr to 29-Apr 2019
        "23-Apr-19", "25-Apr-19", "28-Apr-19", "29-Apr-19",
    ]
    times = [
        " 12:36:55 ", "10:00:00", "15:30:00", "23:59:59",
        "00:00:00", " 23:59:59 ",
        "00:00:00", "12:00:00", "18:00:00", "23:59:58",
    ]
    fc1 = [53287, 41029, 64661, 15879, 38175, 36462, 18658, 40335, 32517, 32861]
    fc2 = [1463, 55891, 40232, 49034, 40957, 11788, 9955, 52379, 32528, 32400]
    fc3 = [33518, 26004, 33460, 60505, 52382, 4504, 53189, 19963, 32676, 32586]
    fc4 = [23014, 50645, 44046, 8898, 216, 55448, 2021, 1888, 32997, 32383]
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  # Only Eval has attack samples
    types = ["normal", "normal", "normal", "normal", "normal", "normal", "scanning", "injection", "backdoor", "normal"]

    df = pd.DataFrame({
        "date": dates,
        "time": times,
        "FC1_Read_Input_Register": fc1,
        "FC2_Read_Discrete_Value": fc2,
        "FC3_Read_Holding_Register": fc3,
        "FC4_Read_Coil": fc4,
        "label": labels,
        "type": types,
    })
    return df


def test_1_exact_feature_ordering(synthetic_modbus_df):
    """Test 1: Verify exact feature columns and deterministic ordering."""
    expected_order = [
        "FC1_Read_Input_Register",
        "FC2_Read_Discrete_Value",
        "FC3_Read_Holding_Register",
        "FC4_Read_Coil",
    ]
    assert TON_IOT_FEATURE_COLUMNS == expected_order

    # Scramble columns in input DataFrame
    scrambled = synthetic_modbus_df[["type", "FC4_Read_Coil", "date", "FC1_Read_Input_Register", "FC3_Read_Holding_Register", "label", "FC2_Read_Discrete_Value", "time"]]
    features = TonIoTDataAdapter.extract_features(scrambled)

    assert list(features.columns) == expected_order
    assert "date" not in features.columns
    assert "time" not in features.columns
    assert "label" not in features.columns
    assert "type" not in features.columns


def test_2_timestamp_reconstruction(synthetic_modbus_df):
    """Test 2: Verify deterministic timestamp parsing from 'date' + 'time'."""
    adapter = TonIoTDataAdapter()
    df_sorted = adapter.reconstruct_timestamps(synthetic_modbus_df)

    assert "timestamp" in df_sorted.columns
    assert pd.api.types.is_datetime64_any_dtype(df_sorted["timestamp"])
    # First row should be 2019-03-31 12:36:55
    assert df_sorted["timestamp"].iloc[0] == pd.Timestamp("2019-03-31 12:36:55")
    # Original row ID must be preserved
    assert "original_row_id" in df_sorted.columns


def test_3_chronological_sorting(synthetic_modbus_df):
    """Test 3: Verify sorting orders out-of-sequence rows correctly."""
    # Reverse the dataframe
    df_reversed = synthetic_modbus_df.iloc[::-1].reset_index(drop=True)
    adapter = TonIoTDataAdapter()
    df_sorted = adapter.reconstruct_timestamps(df_reversed)

    # Assert monotonically increasing timestamps
    assert df_sorted["timestamp"].is_monotonic_increasing


def test_4_negative_time_deltas_and_malformation_detection():
    """Test 4: Verify detection of malformed and missing timestamps."""
    adapter = TonIoTDataAdapter()

    # Case A: Missing required columns
    bad_df1 = pd.DataFrame({"colA": [1, 2]})
    with pytest.raises(ValueError, match="schema violation"):
        adapter.reconstruct_timestamps(bad_df1)

    # Case B: Unparseable date format
    bad_df2 = pd.DataFrame({
        "date": ["invalid-date"],
        "time": ["12:00:00"],
        "FC1_Read_Input_Register": [1],
        "FC2_Read_Discrete_Value": [1],
        "FC3_Read_Holding_Register": [1],
        "FC4_Read_Coil": [1],
        "label": [0],
        "type": ["normal"],
    })
    with pytest.raises(ValueError, match="Malformed timestamps detected"):
        adapter.reconstruct_timestamps(bad_df2)

    # Case C: Null values
    bad_df3 = bad_df2.copy()
    bad_df3["date"] = [np.nan]
    with pytest.raises(ValueError, match="contains missing values"):
        adapter.reconstruct_timestamps(bad_df3)


def test_5_frozen_split_boundaries(synthetic_modbus_df):
    """Test 5: Verify frozen split boundaries and strict temporal precedence."""
    adapter = TonIoTDataAdapter()
    df_sorted = adapter.reconstruct_timestamps(synthetic_modbus_df)

    splitter = TonIoTChronologicalSplitter()
    split = splitter.split(df_sorted)

    # Verify temporal precedence: TRAIN < CALIBRATION < EVALUATION
    assert split.meta_train["timestamp"].max() < split.meta_calib["timestamp"].min()
    assert split.meta_calib["timestamp"].max() < split.meta_eval["timestamp"].min()

    # Verify boundary checks
    assert split.meta_train["timestamp"].min() >= pd.Timestamp("2019-03-31 12:36:52")
    assert split.meta_train["timestamp"].max() <= pd.Timestamp("2019-04-03 23:59:59")

    assert split.meta_calib["timestamp"].min() >= pd.Timestamp("2019-04-04 00:00:00")
    assert split.meta_calib["timestamp"].max() <= pd.Timestamp("2019-04-04 23:59:59")

    assert split.meta_eval["timestamp"].min() >= pd.Timestamp("2019-04-23 00:00:00")
    assert split.meta_eval["timestamp"].max() <= pd.Timestamp("2019-04-29 23:59:58")


def test_6_discrepancy_reporting_on_audited_data_if_available():
    """Test 6: Verify partition counting and discrepancy reporting on audited Modbus CSV."""
    audited_csv = Path(r"C:\Users\Shreyas A\PROJECTS\BAZT-TON_IoT\TON_IoT_audit\iot_processed\IoT_Modbus.csv")
    if not audited_csv.exists():
        pytest.skip(f"Audited CSV not present at {audited_csv}, skipping full dataset verification test.")

    adapter = TonIoTDataAdapter(audited_csv)
    df_raw = adapter.load_raw()
    df_sorted = adapter.reconstruct_timestamps(df_raw)

    splitter = TonIoTChronologicalSplitter()
    split = splitter.split(df_sorted)
    report = split.discrepancy_report

    # Train and Calibration counts must match exactly
    assert len(split.X_train) == 100_940
    assert report["train"]["count_match"] is True
    assert len(split.X_calib) == 11_691
    assert report["calibration"]["count_match"] is True

    # Continuous evaluation count is 173,506 (legacy recommendation was 150,023)
    assert len(split.X_eval) == 173_506
    assert report["evaluation"]["actual_continuous_count"] == 173_506
    assert report["evaluation"]["actual_legacy_count"] == 150_023
    assert report["evaluation"]["attack_samples"] == 64_339
    assert report["temporal_precedence_verified"] is True
    assert report["zero_overlap_verified"] is True


def test_7_training_and_calibration_benign_purity(synthetic_modbus_df):
    """Test 7: Verify training and calibration partitions contain strictly 0 attack samples."""
    adapter = TonIoTDataAdapter()
    df_sorted = adapter.reconstruct_timestamps(synthetic_modbus_df)

    splitter = TonIoTChronologicalSplitter()
    split = splitter.split(df_sorted)

    assert (split.labels_train["label"] == 0).all()
    assert (split.labels_calib["label"] == 0).all()
    assert split.discrepancy_report["train"]["benign_purity_percent"] == 100.0
    assert split.discrepancy_report["calibration"]["benign_purity_percent"] == 100.0


def test_8_scaler_fit_only_on_train(synthetic_modbus_df):
    """Test 8: Verify RobustScaler is fitted strictly on TRAIN and remains frozen."""
    adapter = TonIoTDataAdapter()
    df_sorted = adapter.reconstruct_timestamps(synthetic_modbus_df)

    splitter = TonIoTChronologicalSplitter()
    split = splitter.split(df_sorted)

    preprocessor = TonIoTPreprocessor()
    preprocessor.fit(split.X_train)

    train_medians = preprocessor.center_.copy()
    train_iqrs = preprocessor.scale_.copy()

    # Transform Calib and Eval
    X_calib_scaled = preprocessor.transform(split.X_calib)
    X_eval_scaled = preprocessor.transform(split.X_eval)

    # Assert scaler parameters were not modified during transform
    np.testing.assert_array_equal(preprocessor.center_, train_medians)
    np.testing.assert_array_equal(preprocessor.scale_, train_iqrs)

    # Feature column names must match exactly
    assert list(X_calib_scaled.columns) == TON_IOT_FEATURE_COLUMNS
    assert list(X_eval_scaled.columns) == TON_IOT_FEATURE_COLUMNS


def test_9_labels_not_used_by_preprocessing(synthetic_modbus_df):
    """Test 9: Verify preprocessor forbids label columns to prevent leakage."""
    preprocessor = TonIoTPreprocessor()

    # Pass DataFrame containing 'label' column
    df_with_labels = synthetic_modbus_df[TON_IOT_FEATURE_COLUMNS + ["label"]]
    with pytest.raises(ValueError, match="Protocol violation! Ground truth columns"):
        preprocessor.fit(df_with_labels)

    # Fit properly on features only
    preprocessor.fit(synthetic_modbus_df[TON_IOT_FEATURE_COLUMNS])

    # Attempt to transform with labels
    with pytest.raises(ValueError, match="Protocol violation! Ground truth columns"):
        preprocessor.transform(df_with_labels)


def test_10_edge_iiot_assumptions_absent():
    """Test 10: Verify Edge-IIoTset packet columns are absent from ToN-IoT adapter."""
    adapter = TonIoTDataAdapter()
    # Check required columns
    edge_iiot_cols = ["tcp.len", "ip.src_host", "frame.len", "arp.src.proto_ipv4", "Attack_label", "Attack_type"]
    for col in edge_iiot_cols:
        assert col not in TON_IOT_FEATURE_COLUMNS
        assert col not in TON_IOT_REQUIRED_COLUMNS


def test_11_score_normalization_interface():
    """Test 11: Verify score normalization clipping, bounds, and calibration functions."""
    normalizer = TonIoTScoreNormalizer()

    # Synthetic raw training anomaly scores in range [-0.8, -0.2]
    raw_train = np.array([-0.8, -0.6, -0.5, -0.3, -0.2])
    normalizer.fit(raw_train)

    assert normalizer.s_min_ == pytest.approx(-0.8)
    assert normalizer.s_max_ == pytest.approx(-0.2)

    # Normalizing values inside range
    norm_mid = normalizer.normalize(np.array([-0.5]))
    assert norm_mid[0] == pytest.approx(0.5)

    # Values exceeding bounds must be clipped to [0, 1]
    norm_ext = normalizer.normalize(np.array([-1.2, 0.5]))
    assert norm_ext[0] == 0.0
    assert norm_ext[1] == 1.0

    # Test percentile threshold calculation (method='linear')
    scores_calib = np.linspace(0.0, 1.0, 101)
    a_thr = compute_benign_anomaly_threshold(scores_calib, percentile=99.0)
    assert a_thr == pytest.approx(0.99)

    # Test gamma calculation (mu + 3.05 * sigma)
    ema_calib = np.array([0.10, 0.12, 0.08, 0.10, 0.10])
    gamma, mu, sigma = compute_benign_gamma(ema_calib, multiplier=3.05)
    assert gamma == pytest.approx(mu + 3.05 * sigma)

    # Test simulate_ema_stream
    ema_stream = simulate_ema_stream(np.array([0.5, 0.5, 0.5]), beta=0.40, initial_ema=0.0)
    # Step 0: 0.4 * 0.5 + 0.6 * 0 = 0.2
    assert ema_stream[0] == pytest.approx(0.2)
    # Step 1: 0.4 * 0.5 + 0.6 * 0.2 = 0.32
    assert ema_stream[1] == pytest.approx(0.32)


def test_12_parameter_provenance_and_trust_config_preservation():
    """Test 12: Verify Theorem 1 closed-form alpha and parameter invariance."""
    import importlib.util
    import yaml

    trust_engine_dir = Path(__file__).resolve().parent.parent.parent / "dynamic-trust-engine"
    alpha_path = trust_engine_dir / "src" / "trust_math" / "alpha.py"

    spec = importlib.util.spec_from_file_location("trust_alpha", alpha_path)
    alpha_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alpha_module)
    compute_alpha = alpha_module.compute_alpha

    # Primary configuration: k=10, T_target=0.90 -> alpha approx 0.2057
    alpha_k10 = compute_alpha(target_trust=0.90, recovery_steps=10)
    assert alpha_k10 == pytest.approx(0.20567, abs=1e-4)

    # Secondary sensitivity configuration: k=30, T_target=0.90 -> alpha approx 0.0739
    alpha_k30 = compute_alpha(target_trust=0.90, recovery_steps=30)
    assert alpha_k30 == pytest.approx(0.07398, abs=1e-4)

    # Verify trust config default parameters in YAML directly
    config_yaml = trust_engine_dir / "config" / "trust_config.yaml"
    with open(config_yaml, "r", encoding="utf-8") as f:
        trust_cfg = yaml.safe_load(f)

    assert trust_cfg["slow_burn"]["ema_beta"] == pytest.approx(0.40)
    assert trust_cfg["slow_burn"]["accumulation_step_lambda"] == pytest.approx(0.55)
    assert trust_cfg["slow_burn"]["decay_step_delta"] == pytest.approx(0.05)
    assert trust_cfg["slow_burn"]["trigger_threshold_theta"] == pytest.approx(1.00)
    assert trust_cfg["slow_burn"]["penalty_weight_p"] == pytest.approx(0.25)

    # Verify policy state boundaries
    assert trust_cfg["states"]["HIGH"] == pytest.approx(0.80)
    assert trust_cfg["states"]["MEDIUM"] == pytest.approx(0.60)
    assert trust_cfg["states"]["LOW"] == pytest.approx(0.30)
    assert trust_cfg["states"]["CRITICAL"] == pytest.approx(0.00)
