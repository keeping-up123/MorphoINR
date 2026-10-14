import argparse
import glob
import os
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ===================== 默认只改这里 =====================
#DEFAULT_LOG_DIR = "logs_final/random/Human_Brain_prefrontal_cortex_151509/01_aeonly/lightning_logs/version_0"
DEFAULT_LOG_DIR = "logs_final/random/Human_Brain_prefrontal_cortex_151509/02_aealign/lightning_logs/version_0"
DEFAULT_MONITOR = "val/loss"
# =======================================================

# 只保留你截图对应的12个原始完整tag，顺序和截图一一对应
TARGET_TAGS = [
    "val/mean_absolute_error",
    "val/mean_absolute_error_mask",
    "val/mean_squared_error",
    "val/mean_squared_error_mask",
    "val/cosine_similarity",
    "val/cosine_similarity_mask",
    "val/pearsonr",
    "val/spearmanr",
    "val/pearsonr_mask",
    "val/spearmanr_mask",
    "val/iou",
    "val/sam",
]


def safe_torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def find_best_ckpt(log_dir):
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None

    best_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "best*.ckpt")))
    if len(best_ckpts) == 0:
        return None

    best_ckpts = sorted(best_ckpts, key=os.path.getmtime)
    return best_ckpts[-1]


def nearest_event(events, target_step):
    return min(events, key=lambda e: abs(e.step - target_step))


def main():
    parser = argparse.ArgumentParser(
        description="Read best checkpoint & fixed target metrics, display original full tag name"
    )
    parser.add_argument(
        "--log_dir",
        default=DEFAULT_LOG_DIR,
        help="Lightning log directory, e.g. logs_final/random/<case>/02_aealign/lightning_logs/version_0",
    )
    parser.add_argument(
        "--monitor",
        default=DEFAULT_MONITOR,
        help="Metric used to select best step when checkpoint step is unavailable.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional checkpoint path. If not provided, automatically uses log_dir/checkpoints/best*.ckpt.",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    monitor = args.monitor

    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if len(event_files) == 0:
        raise FileNotFoundError(f"No TensorBoard event files found in: {log_dir}")

    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()
    all_tags = accumulator.Tags()["scalars"]

    if monitor not in all_tags:
        raise KeyError(
            f"Monitor tag '{monitor}' not found.\nAvailable scalar tags:\n{all_tags}"
        )

    ckpt_path = args.ckpt if args.ckpt is not None else find_best_ckpt(log_dir)

    ckpt_epoch = None
    ckpt_global_step = None

    if ckpt_path is not None:
        ckpt = safe_torch_load(ckpt_path)
        ckpt_epoch = ckpt.get("epoch", None)
        ckpt_global_step = ckpt.get("global_step", None)

    # 确定最优step
    if ckpt_global_step is not None:
        best_step = int(ckpt_global_step)
    else:
        monitor_events = accumulator.Scalars(monitor)
        best_event = min(monitor_events, key=lambda e: e.value)
        best_step = best_event.step

    # 精简头部信息
    print("=" * 85)
    print(f"Log Path: {log_dir}")
    print(f"Best Step: {best_step} | Best Epoch: {ckpt_epoch if ckpt_epoch is not None else 'N/A'}")
    if ckpt_path:
        print(f"Best Checkpoint: {os.path.basename(ckpt_path)}")
    print("=" * 85)
    print(f"{'Metric':<45} | {'Value':<15} Step")
    print("-" * 85)

    # 直接遍历原始tag，打印也用完整原始名称，无别名
    for tag in TARGET_TAGS:
        if tag not in all_tags:
            print(f"{tag:<45} | {'N/A':<15} -")
            continue
        events = accumulator.Scalars(tag)
        e = nearest_event(events, best_step)
        print(f"{tag:<45} | {e.value:.8f}  {e.step}")

    print("-" * 85)

    # 可选：列出checkpoint列表，不需要可以整段删除
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        print("\nAll Checkpoints:")
        for ckpt in ckpts:
            print(f"  {ckpt}")


if __name__ == "__main__":
    main()
