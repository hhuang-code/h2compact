from __future__ import annotations

import os
import joblib
from pathlib import Path
from typing import Callable, List, Tuple, Any, Optional
import numpy as np
import bisect
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from data_utils import read_force_torque_txt, parse_local_vel_omega, swt_approx_detail

import pdb


def find_valid_subfolders(root: str | Path) -> List[Path]:
    """
    Traverse `root` recursively and return every directory that:
      • holds ≥1 .mp4 file, and
      • contains a sub‑directory with the same name as (each) .mp4’s stem.

    Examples
    --------
    Suppose dir structure:

        dataset/
        ├─ motion01/
        │  ├─ video01.mp4
        │  └─ video01/          ← matches stem, directory kept
        ├─ motion01/
        │  ├─ video02.mp4
        └─ └─ misc.txt          ← no matching dir → skipped

    """
    root = Path(root).expanduser().resolve()
    valid_dirs: list[Path] = []

    # os.walk would also work, but Path.rglob keeps it tidy.
    for dir_path in (p for p in root.rglob("*") if p.is_dir()):
        # Collect all *.mp4 files directly inside dir_path (no recursion here).
        mp4_files = [f for f in dir_path.iterdir()
                     if f.is_file() and f.suffix.lower() == ".mp4"]

        if not mp4_files:
            continue  # quick reject

        # Verify matching sub‑dirs for every mp4 file present.
        if all((dir_path / f.stem).is_dir() for f in mp4_files):
            valid_dirs.append(dir_path)

    valid_dirs = [valid_dir for valid_dir in valid_dirs if not 'video_without_objects' in str(valid_dir)]

    return valid_dirs


class ForceMovementDataset(Dataset):
    """
    Args
    ----
    data_folders : List[str | Path]
        List of directory that stores the raw data.
    transform : Callable[[Any], Any], optional
        Function applied to the input sample.
    transform_args: dict, optional
        Dictionary of arguments to pass to the function.
    target_transform : Callable[[Any], Any], optional
        Function applied to the target/label.
    target_transform_args: dict, optional
        Dictionary of arguments to pass to the target function.
    """
    def __init__(
        self,
        data_folders: List[str | Path],
        transform: Optional[Callable] = None,
        transform_args: Optional[dict] = None,
        target_transform: Optional[Callable] = None,
        target_transform_args: Optional[dict] = None,
    ):
        self.transform = transform
        self.transform_args = transform_args
        self.target_transform = target_transform
        self.target_transform_args = target_transform_args

        self.force_data1, self.force_data2 = [], []
        self.img_in_force_idx1, self.img_in_force_idx2 = [], []
        self.vel_local, self.omega_yaw = [], []

        self.folders = data_folders

        for folder in self.folders:
            folder_root = Path(folder).expanduser().resolve()

            # Load force data
            txt_files = sorted([p for p in folder_root.rglob("*.txt") if p.is_file() and 'forces' in str(p)])
            try:
                assert len(txt_files) == 2
            except:
                pdb.set_trace()
            force_data1 = read_force_torque_txt(txt_files[0], delimiter=",", skip_lines=1)
            force_data2 = read_force_torque_txt(txt_files[1], delimiter=",", skip_lines=1)

            # Get image names
            img_files = [p for p in folder_root.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"]
            img_names = sorted([float(p.stem) for p in img_files])

            # Get the corresponding force indices of each image
            img_in_force_idx1 = [force_data1['timestep'].index(item) for item in img_names]
            img_in_force_idx2 = [force_data2['timestep'].index(item) for item in img_names]

            # Load movement data
            path = Path(os.path.join(folder_root, 'video_without_objects', 'wham_output.pkl')).expanduser()
            with path.open("rb") as f:
                output = joblib.load(f)

            for k, v in output.items():
                if len(v) > 0 and 'pose_world' in v.keys() and 'trans_world' in v.keys():
                    orient, pos = v['pose_world'][:, :3], v['trans_world']  # both: (n_frames, 3)
                    if not (orient.shape[0] == pos.shape[0] and orient.shape[0] == len(img_names)):
                        continue
                    else:
                        break

            try:
                assert orient.shape[0] == pos.shape[0] and orient.shape[0] == len(img_names) # todo: need a better way
            except:
                continue

            vel_local, omega_yaw = self.target_transform(orient, pos, self.target_transform_args)  # (n_frames - 1, 3), (n_frames - 1,)

            self.force_data1.append(force_data1)
            self.force_data2.append(force_data2)

            self.img_in_force_idx1.append(img_in_force_idx1)
            self.img_in_force_idx2.append(img_in_force_idx2)

            self.vel_local.append(vel_local)
            self.omega_yaw.append(omega_yaw)

        horizon = self.transform_args['horizon']
        self.lens = [len(vel) - 2 * (horizon - 1) for vel in self.vel_local]

    def __len__(self) -> int:
        return sum(self.lens)

    def __getitem__(self, idx: int):
        """Load a single sample *lazily*.

        Returns:
            apx: (H * forces_per_move, level, 6), detl: (H * forces_per_move, level, 6),
            y_vel_local: (H, 3), omega_yaw: (H,)
        """
        cumsum_lens = np.cumsum(self.lens)
        file_idx = bisect.bisect_left(cumsum_lens, idx + 1)

        force_data1, force_data2 = self.force_data1[file_idx], self.force_data2[file_idx]
        img_in_force_idx1, img_in_force_idx2 = self.img_in_force_idx1[file_idx], self.img_in_force_idx2[file_idx]
        vel_local, omega_yaw = self.vel_local[file_idx], self.omega_yaw[file_idx]

        horizon = self.transform_args['horizon']
        t0 = idx - (cumsum_lens[file_idx - 1] if file_idx > 0 else 0) + (horizon - 1)   # starting index of movement

        # Get movement data
        y_vel_local = torch.from_numpy(vel_local[t0:t0 + horizon]).float()  # (horizon, 3)
        y_omega_yaw = torch.from_numpy(omega_yaw[t0:t0 + horizon]).float()  # (horizon,)

        # Get force data
        t0_force_idx1, t1_force_idx1 = img_in_force_idx1[t0 + 1 - horizon], img_in_force_idx1[t0 + 1]
        t0_force_idx2, t1_force_idx2 = img_in_force_idx2[t0 + 1 - horizon], img_in_force_idx2[t0 + 1]

        assert t0_force_idx1 >= 0 and t0_force_idx2 >= 0

        x_force_f1 = np.stack([force_data1['fx'][t0_force_idx1:t1_force_idx1 + 1],
                               force_data1['fy'][t0_force_idx1:t1_force_idx1 + 1],
                               force_data1['fz'][t0_force_idx1:t1_force_idx1 + 1]], axis=1)
        x_force_t1 = np.stack([force_data1['tx'][t0_force_idx1:t1_force_idx1 + 1],
                               force_data1['ty'][t0_force_idx1:t1_force_idx1 + 1],
                               force_data1['tz'][t0_force_idx1:t1_force_idx1 + 1]], axis=1)

        x_force_f2 = np.stack([force_data2['fx'][t0_force_idx2:t1_force_idx2 + 1],
                               force_data2['fy'][t0_force_idx2:t1_force_idx2 + 1],
                               force_data2['fz'][t0_force_idx2:t1_force_idx2 + 1]], axis=1)
        x_force_t2 = np.stack([force_data2['tx'][t0_force_idx2:t1_force_idx2 + 1],
                               force_data2['ty'][t0_force_idx2:t1_force_idx2 + 1],
                               force_data2['tz'][t0_force_idx2:t1_force_idx2 + 1]], axis=1)

        # Upsample force data
        N = x_force_f1.shape[0]

        # RNG setup (NumPy >= 1.17)
        seed = 42
        rng = np.random.default_rng(seed)

        # Build a *balanced* list of indices
        M = self.transform_args['forces_per_move'] * horizon

        if M <= N:
            # Perfectly uniform: choose M distinct indices.
            idx = sorted(rng.choice(N, size=M, replace=False))
        else:
            # Every index appears `q` times, and `r` indices appear one extra time.
            q, r = divmod(M, N)  # M = q*N + r, with r < N
            base = np.repeat(np.arange(N), q)  # each index repeated q times

            # Choose `r` distinct indices to receive one additional copy.
            extras = rng.choice(N, size=r, replace=False)
            idx = sorted(np.concatenate([base, extras]))

        # Gather force data
        x_force_f1, x_force_t1, x_force_f2, x_force_t2 = \
            x_force_f1[idx], x_force_t1[idx], x_force_f2[idx], x_force_t2[idx]

        target_len = 2 ** int(np.ceil(np.log2(M)))  # Find the smallest x such that 2**x >= M
        pad_len = target_len - M  # Calculate required padding length

        x_force_f1 = np.concatenate([np.zeros((pad_len, x_force_f1.shape[1])), x_force_f1])
        x_force_t1 = np.concatenate([np.zeros((pad_len, x_force_t1.shape[1])), x_force_t1])
        x_force_f2 = np.concatenate([np.zeros((pad_len, x_force_f2.shape[1])), x_force_f2])
        x_force_t2 = np.concatenate([np.zeros((pad_len, x_force_t2.shape[1])), x_force_t2])

        x_force = np.concatenate([x_force_f1, x_force_t1, x_force_f2, x_force_t2], axis=1)

        # wavelet decomposition
        WAVELET = self.transform_args['wavelet']
        level = self.transform_args['level']

        apx, detl = self.transform(x_force, WAVELET=WAVELET, level=level)

        apx, detl = apx[pad_len:, ...], detl[pad_len:, ...]     # remove padded data

        apx_force_f1, apx_force_t1, apx_force_f2, apx_force_t2 = torch.split(apx, 3, dim=2)
        detl_force_f1, detl_force_t1, detl_force_f2, detl_force_t2 = torch.split(detl, 3, dim=2)

        apx_f = torch.cat([apx_force_f1, apx_force_f2], dim=-1)
        apx_t = torch.cat([apx_force_t1, apx_force_t2], dim=-1)

        detl_f = torch.cat([detl_force_f1, detl_force_f2], dim=-1)
        detl_t = torch.cat([detl_force_t1, detl_force_t2], dim=-1)

        return apx_f, apx_t, detl_f, detl_t, y_vel_local, y_omega_yaw


if __name__ == "__main__":
    data_root = Path("../extracted_data/ex_data")

    transform_args = {'horizon': 6,
                      'forces_per_move': 33,
                      'wavelet': 'db4',
                      'level': 5}

    target_transform_args = {'dt': 1.0 / 30,  # 1.0 / fps
                             'filter_type': 'uniform',   # uniform or gaussian
                             'kernel_size': 5}

    data_folders = find_valid_subfolders(data_root)

    val_ratio = 0.2
    seed = 42
    train_folders, val_folders = train_test_split(
        data_folders,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    tr_dataset = ForceMovementDataset(
        data_folders=train_folders,
        transform=swt_approx_detail,
        transform_args=transform_args,
        target_transform=parse_local_vel_omega,
        target_transform_args=target_transform_args,
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=32,
        shuffle=True,            # True for training
        num_workers=0,           # <‑‑ tune based on CPU cores / I/O
        pin_memory=True,         # speeds up GPU transfer
        drop_last=False,         # True if you need equal‑sized batches
        persistent_workers=False, # keeps workers alive between epochs (≥ PyTorch 1.9)
    )

    val_dataset = ForceMovementDataset(
        data_folders=val_folders,
        transform=swt_approx_detail,
        transform_args=transform_args,
        target_transform=parse_local_vel_omega,
        target_transform_args=target_transform_args,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,  # True for training
        num_workers=0,  # <‑‑ tune based on CPU cores / I/O
        pin_memory=True,  # speeds up GPU transfer
        drop_last=False,  # True if you need equal‑sized batches
        persistent_workers=False,  # keeps workers alive between epochs (≥ PyTorch 1.9)
    )

    # Example training loop stub
    for epoch in range(1):
        for batch_idx, (apx_f, apx_t, _, _, vel, omega) in enumerate(tr_loader):
            # apx : [B, H * forces_per_move, L, D], vel: (B, H, 3), omega: (B, H)
            # Forward pass, loss, backward, optimizer.step(), etc.

            print(apx_f.shape, vel.shape, omega.shape)
