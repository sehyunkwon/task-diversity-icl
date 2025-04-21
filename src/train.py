from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import copy
import json
import os
import random
import statistics
import sys
import time
import uuid
from collections import deque
from typing import List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import yaml
import wandb
from tqdm import tqdm

# ── Project-specific ──────────────────────────────────────────────────────────
from src.args import build_parser
from src.attention_analysis import nn_scoring_step, prefix_scoring_step
from src.bool_task import *  
from src.continuous_task import * 
from src.curriculum import Curriculum
from src.eval import get_run_metrics
from src.logger import log_metrics
from src.models import build_model
from src.ncl import (
    get_optimal_ncl_acc,
    get_optimal_ncl_bool_loss,
    get_optimal_ncl_bool_loss_from_true_no_context_function,
    get_optimal_ncl_loss,
)
from src.ncl_check import run_ncl_check
from src.optimizer import build_optimizer
from src.remove_pt import delete_pt_files
from src.samplers import get_continuous_data_sampler
from src.setup import setup_device_and_model
from src.task_sampler import build_task_train_list
from src.trunk import update_namespace_with_json
from src.utils import model_dist, model_sim

# ── cuDNN settings ────────────────────────────────────────────────────────────
# Enable benchmark mode to let cuDNN find the best convolution algorithms
# for current architecture and input sizes (use with caution for variable shapes).
torch.backends.cudnn.benchmark = True


# ════════════════════════════════════════════════════════════════════════════
# Training utilities
# ════════════════════════════════════════════════════════════════════════════

def train_step(
    task_names: List[str],
    model: torch.nn.Module,
    xs_list: List[torch.Tensor],
    ys_list: List[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_funcs: List,
    task_train_list,
    precision: str = "full",
) -> Tuple[List[float], List[torch.Tensor], List[float]]:
    """Run a single optimization step over *all* tasks.

    Each task receives its own batch; gradients are accumulated jointly and a
    single ``optimizer.step`` is performed.

    Args:
        task_names: Identifiers for each task in *this* batch (order-aligned with
            ``xs_list`` etc.).
        model: Model under training.
        xs_list: List of input batches per task *(B, …)*.
        ys_list: List of target batches per task *(B, …)*.
        optimizer: Torch optimizer instance.
        loss_funcs: Per-task loss functions.
        task_train_list: Helper objects providing task-specific utilities.
        precision: When set to ``"half"`` targets are down-cast to ``float16`` to
            match mixed-precision training.

    Returns:
        ``loss_list`` - scalar loss per task (already detached).
        ``output_list`` - raw model outputs per task (detached).
        ``last_point_loss_list`` - loss on the final time-step / point per task.
    """
    optimizer.zero_grad()

    cumulative_loss = torch.tensor(0.0, device=model.device)
    loss_list: List[float] = []
    last_point_loss_list: List[float] = []
    output_list: List[torch.Tensor] = []

    for xs, ys, loss_fn, task_train, task_name in zip(
        xs_list, ys_list, loss_funcs, task_train_list, task_names
    ):
        if torch.isnan(xs).any() or torch.isnan(ys).any():
            raise ValueError(f"Detected NaNs in tensors for task: {task_name}")

        outputs = model(xs, ys)

        if precision == "half":
            ys = ys.to(dtype=torch.float16)

        loss = loss_fn(outputs, ys)
        last_point_loss = loss_fn(outputs[:, -1], ys[:, -1])

        # Normalise losses by optimal NCL values when available.
        if task_train.ncl_opt_loss is not None:
            loss /= task_train.ncl_opt_loss
            last_point_loss /= task_train.ncl_opt_loss

        cumulative_loss += loss

        # Bookkeeping for logging
        loss_list.append(loss.item())
        last_point_loss_list.append(last_point_loss.item())
        output_list.append(outputs.detach())

    cumulative_loss.backward()
    optimizer.step()

    return loss_list, output_list, last_point_loss_list


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train(model: torch.nn.Module, args) -> None:
    """Full training routine including curriculum learning and logging."""

    optimizer = build_optimizer(model, args)
    curriculum = Curriculum(args)
    task_list = args.task_list
    device = model.device

    # ── Optional: load pretrained weights ────────────────────────────────────
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained), strict=False)

    # Snapshot of initial weights for distance metrics
    init_model = copy.deepcopy(model)

    # ── Resume from checkpoint if present ────────────────────────────────────
    start_step = 0
    ckpt_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["train_step"]
        # Fast-forward curriculum schedule
        for _ in range(start_step + 1):
            curriculum.update()

    # ── Data sampler setup ───────────────────────────────────────────────────
    n_dims = model.module.n_dims if isinstance(model, torch.nn.DataParallel) else model.n_dims
    data_sampler = get_continuous_data_sampler("gaussian", n_dims=n_dims)

    # ── Rolling windows for metrics ──────────────────────────────────────────
    train_loss_window = {
        task["exp_name"]: deque([float("inf")] * 100, maxlen=100) for task in task_list
    }
    test_loss_window = {
        task["exp_name"]: deque([float("inf")] * 100, maxlen=100)
        for task in task_list
        if task["data"] == "gaussian"
    }
    test_acc_window = {
        task["exp_name"]: deque([0] * 100, maxlen=100)
        for task in task_list
        if task["data"] == "boolean"
    }

    # Flags to log plateau / completion once per task
    plateau_logged = {task["exp_name"]: False for task in task_list}
    complete_logged = {task["exp_name"]: False for task in task_list}

    progress = tqdm(range(start_step, args.train_steps), desc="Training")

    for step in progress:
        task_train_list = build_task_train_list(task_list, n_dims, curriculum)

        """
        No-context Learning (NCL)
        1. run_ncl_check computes the loss value when performing no-context learning (NCL) for each task.
        2. Why is run_ncl_check necessary? Because we need to equalize the loss scale across tasks in order to enable proper analysis. See Section 4.1 and Appendix A for details.
        """

        run_ncl_check(
            step_idx=step,
            args=args,
            task_list=task_list,
            task_train_list=task_train_list,
            data_sampler=data_sampler,
            curriculum=curriculum,
        )

        # ── Build mini-batches ───────────────────────────────────────────────
        xs_list, ys_list, loss_funcs, task_names = [], [], [], []

        for task_cfg, task_train in zip(task_list, task_train_list):
            task_names.append(task_cfg["task"])

            if "bool" not in task_cfg["data"]:
                xs = data_sampler.sample_xs(
                    curriculum.n_points,
                    task_cfg["batch_size"],
                    curriculum.n_dims_truncated,
                )
            else:
                xs = task_train.sample_xs(curriculum.n_points, task_cfg["batch_size"])

            xs_list.append(xs.to(device))
            ys_list.append(task_train.evaluate(xs).to(device))
            loss_funcs.append(task_train.get_training_metric())

        # ── Forward/backward pass ────────────────────────────────────────────
        (loss_list, output_list, last_point_loss_list) = train_step(task_names, model, xs_list, ys_list, optimizer, loss_funcs, task_train_list, precision=args.precision)

        # ── Accuracy / metric computation ───────────────────────────────────
        mean_acc_list, last_acc_list = [], []
        for outputs, ys, task_train in zip(output_list, ys_list, task_train_list):
            metric_fn = task_train.get_metric()

            # mse-style metrics do not produce accuracies
            if metric_fn.__name__ == "squared_error":
                mean_acc_list.append(0.0)
                last_acc_list.append(0.0)
                continue

            pointwise_metric = metric_fn(outputs, ys.to(device)).mean(dim=0)
            mean_acc_list.append(pointwise_metric.mean().item())
            last_acc_list.append(pointwise_metric[-1].item())

        # ── Logging ─────────────────────────────────────────────────────────
        if args.wandb:
            init_dist = model_dist(curr_model=model, init_model=init_model, weight_only=True)
            log_metrics(
                step=step,
                args=args,
                task_list=task_list,
                train_loss_list_of_list=train_loss_window,
                test_loss_list_of_list=test_loss_window,
                test_acc_list_of_list=test_acc_window,
                mean_acc_list=mean_acc_list,
                last_acc_list=last_acc_list,
                loss_list=loss_list,
                last_point_loss_list=last_point_loss_list,
                plateau_logged=plateau_logged,
                complete_logged=complete_logged,
                init_distance=init_dist,
            )

        # ── Checkpointing ───────────────────────────────────────────────────
        if args.save_iteration is not None and step == args.save_iteration:
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.name}_model_{step}.pt"))

        curriculum.update()
        progress.set_description(f"loss {sum(loss_list):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# Entry point helpers
# ════════════════════════════════════════════════════════════════════════════

def main(args) -> None:
    """Script entry: sets up logging, devices, and launches training."""

    if args.test_run:
        # Accelerate dev runs by shrinking curriculum and step count
        args.curriculum_points_start = args.curriculum_points_end
        args.curriculum_dims_start = args.curriculum_dims_end
        args.train_steps = 100
    else:
        # Ensure curriculum spans the full dimensionality during real runs
        args.curriculum_dims_end = args.n_dims
        args.curriculum_dims_start = args.curriculum_dims_end

        if args.wandb:
            wandb.init(
                dir=args.out_dir,
                project=args.project,
                entity=args.entity,
                config=args.__dict__,
                notes=args.notes,
                name=args.name,
                resume=True,
            )

    # Device + model setu
    device, model = setup_device_and_model(args)

    # ── Training ────────────────────────────────────────────────────────────
    train(model, args)

    # ── Evaluation ──────────────────────────────────────────────────────────
    if not args.test_run:
        eval_metrics = get_run_metrics(args.out_dir, device=device)["standard"]
        val_acc = eval_metrics[
            model.module.name if isinstance(model, torch.nn.DataParallel) else model.name
        ]["mean"]
        mean_val_acc = float(np.mean(val_acc))

        if args.wandb:
            wandb.log({"mean_val_acc": mean_val_acc})

            # Prepare line-series plot
            series = [m["mean"] for m in eval_metrics.values()]
            x_axis = list(range(len(series[0])))
            wandb.log(
                {
                    "eval/mean_acc": wandb.plot.line_series(
                        x_axis,
                        series,
                        keys=list(eval_metrics.keys()),
                        title="Accuracy over In-context Examples",
                        xname="In-context Examples",
                    )
                }
            )

    # ── Cleanup ────────────────────────────────────────────────────────────
    if args.delete:
        print("Deleting *.pt model files…")
        delete_pt_files(args.out_dir)


# ════════════════════════════════════════════════════════════════════════════
# CLI wrapper
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()

    # Override args with JSON config when provided
    if cli_args.json_file_path:
        cli_args = update_namespace_with_json(cli_args)

    # ── Output directory & run naming ───────────────────────────────────────
    if not cli_args.test_run:
        run_id = cli_args.resume_id or str(uuid.uuid4())[:20]
        cli_args.name += f"_{cli_args.family}"
        if cli_args.family in {"gpt", "mysan", "attclf"}:
            cli_args.name += f"_{cli_args.model_name}"
        cli_args.name += f"_{run_id[:8]}"

        task_suffix = "+".join(task["task"] for task in cli_args.task_list)
        cli_args.out_dir = os.path.join(cli_args.out_dir, task_suffix, cli_args.name)
        os.makedirs(cli_args.out_dir, exist_ok=True)

        # Store config for reproducibility
        with open(os.path.join(cli_args.out_dir, "config.yaml"), "w") as fh:
            yaml.safe_dump(cli_args.__dict__, fh)

    main(cli_args)
