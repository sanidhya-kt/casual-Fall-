import numpy as np

def compute_smv(acc_data: np.ndarray) -> np.ndarray:
    """
    Computes the Sum Magnitude Vector (SMV) for tri-axial acceleration signals.
    [PAPER SPECIFICATION] Equation (1):
        SMV_t = sqrt((x_t^{AX})^2 + (x_t^{AY})^2 + (x_t^{AZ})^2)
    
    Args:
        acc_data: np.ndarray of shape (T, 3) or (3, T) representing the 3-axis accelerometer data (X, Y, Z).
    
    Returns:
        smv: np.ndarray of shape (T,) containing the SMV at each time step.
    """
    if acc_data.ndim != 2:
        raise ValueError(f"Expected 2D acceleration array, got shape {acc_data.shape}")
    
    if acc_data.shape[0] == 3 and acc_data.shape[1] != 3:
        # Shape is (3, T), transpose to (T, 3)
        acc_data = acc_data.T
        
    if acc_data.shape[1] < 3:
        raise ValueError(f"Expected at least 3 acceleration axes, got {acc_data.shape[1]}")
    
    # Take first 3 axes (X, Y, Z)
    ax, ay, az = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
    smv = np.sqrt(ax**2 + ay**2 + az**2)
    return smv


def detect_impact_time(acc_data: np.ndarray) -> int:
    """
    Determines the impact moment t_impact from the maximum SMV.
    [PAPER SPECIFICATION] Section 3, Phase 1:
        t_impact = argmax_t { SMV(t) }
    
    Args:
        acc_data: np.ndarray of shape (T, 3) or (3, T)
        
    Returns:
        t_impact: int index at which SMV is maximized.
    """
    smv = compute_smv(acc_data)
    t_impact = int(np.argmax(smv))
    return t_impact
