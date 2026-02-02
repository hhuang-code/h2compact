from typing import List, Dict, Optional
import numpy as np
import pywt
import torch

import pdb


from preprocess.odometry import (axis_angle_to_rotation_matrix_np,
                                 rotation_matrix_to_axis_angle_np,
                                 smooth_rotation_matrix_sequence,
                                 convert_velocity_global_to_local,
                                 axis_angle_F1_to_F2,
                                 angular_velocity_from_axis_angle,
                                 rotation_matrix_to_euler_xyz_np)

import pdb


def read_force_torque_txt(path: str,
                          delimiter: Optional[str] = None,
                          skip_lines: int = 0) -> Dict[str, List[float]]:
    """
    Read a text file whose first seven columns are: timestep, fx, fy, fz, tx, ty, tz.

    Parameters
    ----------
    path : str
        Path to the .txt file.
    delimiter : str or None, optional
        Field delimiter passed to str.split(). None (default) means any
        consecutive whitespace.
    skip_lines : int, default 0
        Number of initial lines to skip (e.g., header).

    Returns
    -------
    dict[str, list[float]]
        Dictionary with keys 'timestep', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'.
    """
    keys = ('timestep', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz')
    data = {k: [] for k in keys}

    with open(path, 'r') as f:
        # Skip user‑specified header lines
        for _ in range(skip_lines):
            next(f, None)

        for line in f:
            line = line.strip()

            parts = line.split(delimiter)
            if len(parts) < 7:
                raise ValueError(f"Line has fewer than 7 columns: {line!r}")

            # Convert the first seven fields to float and append
            for key, value in zip(keys, map(float, parts[:7])):
                data[key].append(value)

    return data


def parse_local_vel_omega(orient, pos, args):
    """
    Args:
        orient: (N, 3)
        pos: (N, 3)
        args: a dictionary

    Returns: vel: (N - 1, 3), omega: (N - 1,)
    """

    dt = args['dt']
    filter_type = args['filter_type']
    kernel_size = args['kernel_size']

    R_global = axis_angle_to_rotation_matrix_np(orient)  # (N, 3, 3)
    R_global = smooth_rotation_matrix_sequence(R_global, filter_type=filter_type,
                                               kernel_size=kernel_size)  # smoothing rotation
    aa_global_axis, aa_global_mag, aa_global = rotation_matrix_to_axis_angle_np(R_global)  # unit vec, mag, axis_angle

    # ∆translation in the global frame
    d_global = pos[1:] - pos[:-1]  # (N - 1, 3)

    # Compute velocities
    dt_arr = np.broadcast_to(dt, (d_global.shape[0],)).astype(float)
    v_global = d_global / dt_arr[:, None]

    vel_local = convert_velocity_global_to_local(v_global, aa_global[1:])  # (N - 1, 3)

    # aa_local = axis_angle_F1_to_F2(aa_global)

    omega_global = angular_velocity_from_axis_angle(aa_global, dt)
    # omega_local = angular_velocity_from_axis_angle(aa_local, dt)
    # omega_global = omega_local

    omega_R_global = axis_angle_to_rotation_matrix_np(omega_global)  # (N, 3, 3)
    # omega_R_global = smooth_rotation_matrix_sequence(omega_R_global, filter_type=filter_type, kernel_size=kernel_size)  # smoothing rotation
    # _, _, _omega_global = rotation_matrix_to_axis_angle_np(omega_R_global)

    omega_global = rotation_matrix_to_euler_xyz_np(omega_R_global)

    omega_yaw = omega_global[:, 1]  # (N - 1,)

    return vel_local, omega_yaw


def swt_approx_detail(block, WAVELET='db4', level=5):
    """
    Args:
        block: (T, D)
        WAVELET: str, wavelet type
        level: int, decomposition scales

    Returns: (T, L, D)
    """
    z_a, z_d = [], []

    for c in range(block.shape[1]):
        coeffs = pywt.swt(block[:, c], WAVELET, level=level, trim_approx=False) # [(cAn, cDn), ..., (cA2, cD2), (cA1, cD1)]
        c_d, c_a = [], []
        for idx, (cA, cD) in enumerate(reversed(coeffs)):
            c_a.append(cA)
            c_d.append(cD)
        z_a.append(np.stack(c_a, axis=-1))  # each elem: (T, L)
        z_d.append(np.stack(c_d, axis=-1))

    z_a = np.stack(z_a, axis=-1)    # (T, L, D)
    z_d = np.stack(z_d, axis=-1)

    return torch.from_numpy(z_a).float(), torch.from_numpy(z_d).float()
