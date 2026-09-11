import os
import sys
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.impact_detector import compute_smv, detect_impact_time
from preprocessing.resampler import resample_signal
from preprocessing.windowing import extract_fall_windows, extract_adl_windows
from preprocessing.normalizer import ChannelNormalizer

def test_smv_and_impact():
    # Synthetic acceleration data: 100 timesteps, peak at t=60
    t = np.linspace(0, 4, 100)
    ax = np.sin(t)
    ay = np.cos(t)
    az = np.zeros_like(t)
    # Inject large impact peak at index 60
    ax[60] = 10.0
    ay[60] = 10.0
    az[60] = 10.0
    
    acc_data = np.stack([ax, ay, az], axis=1) # (100, 3)
    smv = compute_smv(acc_data)
    assert smv.shape == (100,), f"Expected shape (100,), got {smv.shape}"
    expected_peak = np.sqrt(10.0**2 + 10.0**2 + 10.0**2)
    assert np.isclose(smv[60], expected_peak), f"Expected peak {expected_peak}, got {smv[60]}"
    
    t_impact = detect_impact_time(acc_data)
    assert t_impact == 60, f"Expected t_impact == 60, got {t_impact}"
    print("[PASS] test_smv_and_impact passed.")


def test_resampler():
    # SisFall: 200 Hz -> 25 Hz (duration = 2 seconds, 400 samples -> 50 samples)
    data_200hz = np.random.randn(400, 9)
    res_25hz = resample_signal(data_200hz, orig_sr=200, target_sr=25)
    assert res_25hz.shape == (50, 9), f"Expected shape (50, 9), got {res_25hz.shape}"
    
    # KFall: 100 Hz -> 25 Hz (duration = 2 seconds, 200 samples -> 50 samples)
    data_100hz = np.random.randn(200, 9)
    res_25hz_kfall = resample_signal(data_100hz, orig_sr=100, target_sr=25)
    assert res_25hz_kfall.shape == (50, 9), f"Expected shape (50, 9), got {res_25hz_kfall.shape}"
    print("[PASS] test_resampler passed.")


def test_windowing():
    # Create 25Hz data of length 200 timesteps (8 seconds)
    data = np.random.randn(200, 9)
    # Impact at t=120
    data[120, :3] = 20.0
    
    delta = 72
    theta = 15
    
    x_falling, y_impact, t_impact = extract_fall_windows(
        data,
        acc_indices=(0, 1, 2),
        delta=delta,
        theta=theta,
        extract_impact=True
    )
    
    assert t_impact == 120, f"Expected impact at 120, got {t_impact}"
    assert x_falling.shape == (9, delta), f"Expected falling shape (9, {delta}), got {x_falling.shape}"
    assert y_impact.shape == (9, delta), f"Expected impact shape (9, {delta}), got {y_impact.shape}"
    
    # Check exact indices:
    # L_falling = [120 - 15 - 72, 120 - 15) = [33, 105)
    # L_impact = [120, 120 + 72) = [120, 192)
    expected_falling = data[33:105, :].T
    assert np.allclose(x_falling, expected_falling), "Falling window content does not match expected slice!"
    
    expected_impact = data[120:192, :].T
    assert np.allclose(y_impact, expected_impact), "Impact window content does not match expected slice!"
    
    # ADL peak extraction
    adl_wins = extract_adl_windows(data, acc_indices=(0, 1, 2), delta=delta, theta=theta, mode='peak')
    assert len(adl_wins) == 1
    assert adl_wins[0].shape == (9, delta)
    
    print("[PASS] test_windowing passed.")


def test_normalizer():
    # Shape (N, C, delta) = (10, 9, 72)
    data = np.random.normal(loc=5.0, scale=2.0, size=(10, 9, 72))
    norm = ChannelNormalizer()
    normed = norm.fit_transform(data)
    
    assert normed.shape == (10, 9, 72)
    # Check mean across (N, delta) per channel is ~0 and std is ~1
    channel_means = np.mean(normed, axis=(0, 2))
    channel_stds = np.std(normed, axis=(0, 2))
    assert np.allclose(channel_means, 0.0, atol=1e-5), f"Means not near 0: {channel_means}"
    assert np.allclose(channel_stds, 1.0, atol=1e-5), f"Stds not near 1: {channel_stds}"
    print("[PASS] test_normalizer passed.")


if __name__ == '__main__':
    test_smv_and_impact()
    test_resampler()
    test_windowing()
    test_normalizer()
    print("\nALL PREPROCESSING TESTS PASSED SUCCESSFULLY!")
