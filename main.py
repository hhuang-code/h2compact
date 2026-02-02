# Complete implementation: Multi-scale diffusion policy
# Conditioned on wavelet approximations (A) and RMS (r)
# Training and inference (DDIM) included

import argparse, glob, os, time

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from transformer import MultiScaleTransformer
from stoch_transformer import StochMultiScaleTransformer
from dataset import ForceMovementDataset, find_valid_subfolders

from data_utils import parse_local_vel_omega, swt_approx_detail

from typing import Dict, List, Tuple
import threading

import rospy
from geometry_msgs.msg import WrenchStamped, Twist

import wandb

import pdb


# ─────────────────────────── Runtime config ────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """
    Fix all major RNG sources (Python, NumPy, PyTorch) so experiments
    are repeatable across runs—CPU and GPU.

    Parameters
    ----------
    seed : int
        The seed value to use for every RNG.
    """
    # --- Python's built‑in RNG -------------------------------------------
    random.seed(seed)

    # --- NumPy -----------------------------------------------------------
    np.random.seed(seed)

    # --- PyTorch: CPU & (all) GPU(s) -------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # if you have multi‑GPU

    # --- (Optional) Configure deterministic CuDNN ------------------------
    #   Deterministic algorithms trade speed for exact reproducibility.
    #   Leave these two lines out if you prefer maximum performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # --- (Optional) Make hash‑based ops deterministic --------------------
    os.environ["PYTHONHASHSEED"] = str(seed)


class MultiScalePolicy(nn.Module):
    def __init__(self, model, args):
        super().__init__()

        if model == "transformer":
            self.policy = MultiScaleTransformer(
                S=args.forces_per_move,
                H=args.horizon,
                act_dim=args.act_dim,
                force_dim=args.force_dim,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                drop=args.drop,
            )

        elif model == "stoch_transformer":
            self.policy = StochMultiScaleTransformer(
                S=args.forces_per_move,
                H=args.horizon,
                act_dim=args.act_dim,
                force_dim=args.force_dim,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                drop=args.drop,
            )
        else:
            raise NotImplementedError(f'{args.model} is not implemented.')

    def forward(self, noisy, t, apx_f, apx_t):
        """
        Args:
            noisy: (B, H, act_dim)
            t: (B,)
            apx_f and apx_t: (B, T=2**x, L, force_dim), L is wavelet levels

        Returns: (B, H, act_dim)
        """

        eps = self.policy(noisy, t, apx_f, apx_t)

        return eps


# ─────────────────────────────  Training  ──────────────────────────────────
class Trainer:
    def __init__(self, policy, betas, lr=1e-3, w_kl=0.01, model='transformer'):
        self.betas = betas.to(DEVICE)
        self.policy = policy.to(DEVICE)
        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.alphas_bar = torch.cumprod(1 - betas, 0).to(DEVICE)

        self.w_kl = w_kl

        self.model = model

    def step(self, batch):
        # apx_f, apx_t: (B, N = H * S, L, D); vel: (B, H, C); omega: (B, H)
        apx_f, apx_t, _, _, vel, omega = batch
        apx_f, apx_t, vel, omega = apx_f.to(DEVICE), apx_t.to(DEVICE), vel.to(DEVICE), omega.to(DEVICE)

        y = torch.cat([vel, omega[..., None]], dim=-1)

        B = y.size(0)
        t = torch.randint(0, len(self.betas), (B,), device=DEVICE)
        alpha_bar = self.alphas_bar[t].view(B, 1, 1)
        eps = torch.randn_like(y)
        noisy = alpha_bar.sqrt() * y + (1 - alpha_bar).sqrt() * eps # (B, H, D), D = C + 1

        if self.model == 'transformer':
            eps_pred = self.policy(noisy, t, apx_f, apx_t)
            eps_loss = F.mse_loss(eps_pred, eps)
            loss = eps_loss
        elif self.model == 'stoch_transformer':
            eps_pred, kl_loss = self.policy(noisy, t, apx_f, apx_t)
            eps_loss = F.mse_loss(eps_pred, eps)
            loss = eps_loss + self.w_kl * kl_loss
        else:
            raise NotImplementedError(f'{args.model} is not implemented.')

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.model == 'transformer':
            return loss.item(), eps_loss.item()
        elif self.model == 'stoch_transformer':
            return loss.item(), eps_loss.item(), kl_loss.item()
        else:
            raise NotImplementedError(f'{args.model} is not implemented.')


# ─────────────────────────────  DDIM sampler  ──────────────────────────────
@torch.no_grad()
def ddim_sample(policy, betas, apx_f, apx_t, steps=20, args=None):
    B = apx_f.size(0)

    alphas_bar = torch.cumprod(1 - betas, 0).to(DEVICE)
    seq = torch.linspace(len(betas) - 1, 0, steps, dtype=torch.long).to(DEVICE)
    y = torch.randn(B, args.horizon, args.act_dim).to(DEVICE)

    for i, t in enumerate(seq):
        t_fill = torch.tensor([t]).repeat((B,)).to(DEVICE)
        out = policy(y, t_fill, apx_f, apx_t)   # (B, H, act_dim)
        eps = out if not isinstance(out, tuple) else out[0]
        alpha_bar = alphas_bar[t]
        if i == steps - 1:
            y = (y - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        else:
            alpha_bar_next = alphas_bar[seq[i + 1]]
            y0 = (y - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
            y = alpha_bar_next.sqrt() * y0 + (1 - alpha_bar_next).sqrt() * eps

    return y


def run_validation_ddim(trainer, val_loader, args, steps=20):  # <<< VAL
    """
    Compute mean MSE between DDIM rollout and ground‑truth actions.
    """
    mse_vals = []

    with torch.no_grad():
        for batch in val_loader:
            apx_f, apx_t, _, _, vel, omega = batch
            apx_f, apx_t = apx_f.to(DEVICE), apx_t.to(DEVICE)
            vel, omega = vel.to(DEVICE), omega.to(DEVICE)
            y_gt = torch.cat([vel, omega[..., None]], dim=-1)

            y_pred = ddim_sample(policy, trainer.betas, apx_f, apx_t, steps, args)  # (B, H, act_dim)

            mse_vals.append(F.mse_loss(y_pred, y_gt, reduction="mean").item())

    return mse_vals


# ─────────────────────────────  ros variables and functions  ──────────────────────────────

# ───────────────────────────────────────────────────────────────────────────
# Shared state (guarded by cache_lock)
# ───────────────────────────────────────────────────────────────────────────
movement_cmd: Dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}
wrench_cache: List[Tuple[str, Tuple[float, ...]]] = []   # [(sensor, 6‑tuple), …]
cache_lock = threading.Lock()


# ───────────────────────────────────────────────────────────────────────────
# Subscriber callback – add each sample to the cache
# ───────────────────────────────────────────────────────────────────────────
def wrench_callback(msg: WrenchStamped, sensor_name: str) -> None:
    fx, fy, fz = msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z
    tx, ty, tz = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z

    with cache_lock:
        wrench_cache.append((sensor_name, (fx, fy, fz, tx, ty, tz)))

    # rospy.loginfo_throttle(
    #     1.0,
    #     "[%s] cached  F=(%.2f,%.2f,%.2f) τ=(%.2f,%.2f,%.2f)   (cache len=%d)",
    #     sensor_name, fx, fy, fz, tx, ty, tz, len(wrench_cache)
    # )

    # rospy.loginfo("[%s] cached  F=(%.2f,%.2f,%.2f) τ=(%.2f,%.2f,%.2f)   (cache len=%d)",
    #     sensor_name, fx, fy, fz, tx, ty, tz, len(wrench_cache))


# ───────────────────────────────────────────────────────────────────────────
# Thread 1 : subscribe & spin
# ───────────────────────────────────────────────────────────────────────────
def force_subscriber_thread() -> None:
    sensor_topics = {
        "mini40": "/ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed",
        "mini45": "/ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed",
    }
    for name, topic in sensor_topics.items():
        rospy.Subscriber(topic, WrenchStamped, wrench_callback, callback_args=name)

    rospy.loginfo("Subscriber thread listening on both ATI sensors.")
    rospy.spin()        # callback handling stays here


def pack_data(batch, args):
    x_force_f1, x_force_t1 = [], []
    x_force_f2, x_force_t2 = [], []

    for data in batch:
        if 'mini40' == data[0]:
            x_force_f1.append(data[1][:3])
            x_force_t1.append(data[1][3:])
        elif 'mini45' == data[0]:
            x_force_f2.append(data[1][:3])
            x_force_t2.append(data[1][3:])
        else:
            raise ValueError(f"Not supported sensor type: {data[0]}")

    x_force_f1 = np.stack(x_force_f1, axis=0)   # (T, 3)
    x_force_t1 = np.stack(x_force_t1, axis=0)   # (T, 3)
    x_force_f2 = np.stack(x_force_f2, axis=0)
    x_force_t2 = np.stack(x_force_t2, axis=0)

    # Upsample force data
    N = min(x_force_f1.shape[0], x_force_f2.shape[0])   # the numbers of force data may differ

    # RNG setup (NumPy >= 1.17)
    seed = 42
    rng = np.random.default_rng(seed)

    # Build a *balanced* list of indices
    M = args.forces_per_move * args.horizon

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
    WAVELET = args.wavelet
    level = args.level

    apx, detl = swt_approx_detail(x_force, WAVELET=WAVELET, level=level)

    apx, detl = apx[pad_len:, ...], detl[pad_len:, ...]  # remove padded data

    apx_force_f1, apx_force_t1, apx_force_f2, apx_force_t2 = torch.split(apx, 3, dim=2)

    apx_f = torch.cat([apx_force_f1, apx_force_f2], dim=-1)
    apx_t = torch.cat([apx_force_t1, apx_force_t2], dim=-1)

    return apx_f, apx_t

# ───────────────────────────────────────────────────────────────────────────
# Thread 2 : publish at 5 Hz
# ───────────────────────────────────────────────────────────────────────────
def publisher_thread(policy, betas, args) -> None:
    pub  = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rate = rospy.Rate(5)      # 5Hz

    rospy.loginfo("Publisher thread started at 5Hz.")

    while not rospy.is_shutdown():
        # ── Copy & clear cache atomically ──────────────────────────────────
        with cache_lock:
            batch = wrench_cache[:]
            wrench_cache.clear()

        if len(batch) > (30 * 6) * 2:

            # ── Run model on the whole batch ───────────────────────────────────
            apx_f, apx_t = pack_data(batch, args)
            apx_f, apx_t = apx_f[None, ...].to(DEVICE), apx_t[None, ...].to(DEVICE)

            y_pred = ddim_sample(policy, betas, apx_f, apx_t, steps=20, args=args)

            new_cmd = {
                "x": y_pred[0, 0, 0].item(),   # only use the first predicted action
                "y": y_pred[0, 0, 1].item(),
                "yaw": 0    # y_pred[0, 0, 3].item()
            }

            movement_cmd.update(new_cmd)

            # ── Publish ────────────────────────────────────────────────────────
            twist = Twist()
            twist.linear.x  = movement_cmd["x"]
            twist.linear.y  = movement_cmd["y"]
            twist.angular.z = movement_cmd["yaw"]
            pub.publish(twist)

            rospy.loginfo(
                "Published cmd: x=%.6f, y=%.6f, yaw=%.6f  (batchsize=%d)",
                movement_cmd["x"],
                movement_cmd["y"],
                movement_cmd["yaw"],
                len(batch)
            )

        rate.sleep()


# ─────────────────────────────  main function  ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_root", default="../extracted_data/ex_data")
    parser.add_argument("--val_ratio", type=float, default=0.2)

    # data params
    parser.add_argument("--horizon", type=int, default=6, help="prediction horizon")
    parser.add_argument("--wavelet", choices=['haar', 'db4', 'sym4', 'coif5'],  default='db4')
    parser.add_argument("--level", type=int, default=5, help="maximum number of wavelet levels")
    parser.add_argument("--forces_per_move", type=int, default=33)
    parser.add_argument("--act_dim", type=int, default=4)
    parser.add_argument("--force_dim", type=int, default=6, help="force dimension per side")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--filter_type", choices=['uniform', 'gaussian'], default='uniform')
    parser.add_argument("--kernel_size", type=int, default=5)

    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--model", choices=["transformer", "stoch_transformer"], default="transformer")

    # model params
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--drop", type=float, default=0.5)

    parser.add_argument("--ddim_steps", type=int, default=20)

    # optimization params
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_workers", type=int, default=4)

    # loss weights
    parser.add_argument("--w_kl", type=float, default=1e-2, help="weight kl divergence")

    # checkpointing
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="model_best.pt")

    # W&B hyper‑params ----------------------------------------------------
    parser.add_argument("--proj_name", default="h2compact", help="W&B project name")
    parser.add_argument("--run_name", default="run_${time}", help="W&B run name")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # W&B init
    wandb.init(project=args.proj_name, name=args.run_name, config={
        **vars(args),  # argparse hyperparams
        "FPS": args.fps,
        "FORCES_PER_MOVE": args.forces_per_move,
        "HORIZON": args.horizon,
        "FORCE_DIM": args.force_dim, "ACT_DIM": args.act_dim,
        "WAVELET": args.wavelet, "LEVEL": args.level,
    }, mode="disabled" if args.mode == "infer" else "online")

    policy = MultiScalePolicy(args.model, args).to(DEVICE)
    wandb.watch(policy, log_graph=False)

    betas = torch.linspace(0.0001, 0.02, 1000).clamp(max=.999)

    if args.mode == "train":
        transform_args = {'horizon': args.horizon,
                          'forces_per_move': args.forces_per_move,
                          'wavelet': args.wavelet,
                          'level': args.level}

        target_transform_args = {'dt': 1.0 / args.fps,
                                 'filter_type': args.filter_type,  # uniform or gaussian
                                 'kernel_size': args.kernel_size}

        data_folders = find_valid_subfolders(args.data_root)

        train_folders, val_folders = train_test_split(
            data_folders,
            test_size=args.val_ratio,
            random_state=args.seed,
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
            batch_size=args.batch,
            shuffle=True,  # True for training
            num_workers=args.n_workers,  # <‑‑ tune based on CPU cores / I/O
            pin_memory=True,  # speeds up GPU transfer
            drop_last=True,  # True if you need equal‑sized batches
            persistent_workers=args.n_workers > 0,  # keeps workers alive between epochs (≥ PyTorch 1.9)
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
            batch_size=args.batch,
            shuffle=False,  # True for training
            num_workers=args.n_workers,  # <‑‑ tune based on CPU cores / I/O
            pin_memory=True,  # speeds up GPU transfer
            drop_last=False,  # True if you need equal‑sized batches
            persistent_workers=args.n_workers > 0,  # keeps workers alive between epochs (≥ PyTorch 1.9)
        )

        trainer = Trainer(policy, betas, args.lr, w_kl=args.w_kl, model=args.model)

        for ep in range(args.epochs):
            if args.model == "transformer":
                policy = trainer.policy
                policy.train()
                losses = [trainer.step(b) for b in tr_loader]
                loss, eps_loss = zip(*losses)
                metrics = {
                    "loss": float(np.mean(loss)),
                    "eps_loss": float(np.mean(eps_loss)),
                }

                policy = trainer.policy
                policy.eval()
                val_mses = run_validation_ddim(trainer, val_loader, args, args.ddim_steps)
                metrics['val_mses'] = float(np.mean(val_mses))

                print(f"Epoch{ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "epoch"))
                wandb.log(metrics)
            elif args.model == 'stoch_transformer':
                losses = [trainer.step(b) for b in tr_loader]
                loss, eps_loss, kl_loss = zip(*losses)
                metrics = {
                    "loss": float(np.mean(loss)),
                    "eps_loss": float(np.mean(eps_loss)),
                    "kl_loss": float(np.mean(kl_loss)),
                }

                policy = trainer.policy
                policy.eval()
                val_mses = run_validation_ddim(trainer, val_loader, args, args.ddim_steps)
                metrics['val_mses'] = float(np.mean(val_mses))

                print(f"Epoch{ep + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "epoch"))
                wandb.log(metrics)
            else:
                raise NotImplementedError(f'{args.model} is not implemented.')

            if args.save:
                torch.save(policy.state_dict(), os.path.join(args.checkpoint_path, args.model + '_' + args.checkpoint_name))

        wandb.finish()
    else:
        if not args.weight:
            raise ValueError("--weight path required for inference mode")

        policy.load_state_dict(torch.load(args.weight, map_location=DEVICE))
        print('Weight loaded: {}\n'.format(args.weight))

        policy.eval()

        rospy.init_node("force_to_action_node", anonymous=True)

        threading.Thread(target=force_subscriber_thread, daemon=True).start()
        threading.Thread(target=publisher_thread, args=(policy, betas, args), daemon=True).start()

        rospy.loginfo("force_to_action_node running (cached batch → model every 0.2s).")
        rospy.spin()

        # while True:
        #     act = ddim_sample(policy, betas, apx)
        #     print("Predicted action:", act.cpu().numpy())
        #     time.sleep(1 / args.fps)
