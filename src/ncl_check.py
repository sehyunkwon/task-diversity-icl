"""
Utility for computing and logging NCL (optimal loss/accuracy) only once.
"""

from collections import deque
import torch
import wandb

from src.ncl import (
    get_optimal_ncl_acc,
    get_optimal_ncl_loss,
    get_optimal_ncl_bool_loss,
)

__all__ = ["run_ncl_check"]


@torch.no_grad()
def run_ncl_check(
    *,
    step_idx: int,
    args,
    task_list: list,
    task_train_list: list,
    data_sampler,
    curriculum,
):
    """Computes NCL values and logs them to wandb only at the first step (i == 0)."""
    if step_idx != 0:
        return

    # Containers to accumulate results
    cont_ys_list, bool_ys_list = [], []
    cont_ncl_loss_list, bool_ncl_acc_list = [], []

    # Iterate through all tasks and compute NCL
    cont_xs = bool_xs = None
    for t, task_train in enumerate(task_train_list):
        task_cfg = task_list[t]

        # ── Continuous ──────────────────────────────────
        if "bool" not in task_cfg["data"]:
            if cont_xs is None:
                cont_xs = data_sampler.sample_xs(
                    args.ncl_bsize, 1, curriculum.n_dims_truncated
                )
            wb = task_train.sample_wb(args.ncl_wsize)
            ys = task_train.evaluate_by_given_wb(cont_xs, wb)  # [w, b]
            indiv_loss = get_optimal_ncl_loss([ys.T])
            cont_ncl_loss_list.append(indiv_loss)
            cont_ys_list.append(ys.T)

            # Cache ReLU, sine, logistic NCL optimal loss into the task
            if any(k in task_cfg["task"] for k in ("relu", "sine", "logistic")):
                task_train.NCL_optimal_loss(indiv_loss)

            if args.wandb:
                wandb.log({f"NCL optimum loss - {task_cfg['exp_name']}": indiv_loss}, step=0)

        # ── Boolean ─────────────────────────────────────
        else:
            if bool_xs is None:
                bool_xs = task_train.sample_xs(args.ncl_bsize, 1)
            wb = task_train.sample_wb(args.ncl_wsize)
            ys = task_train.evaluate_by_given_wb(bool_xs, wb)  # [w, b]
            indiv_acc = get_optimal_ncl_acc([ys.T])
            indiv_loss = get_optimal_ncl_bool_loss(task_cfg["exp_name"])

            task_train.NCL_optimal_loss(indiv_loss)
            bool_ncl_acc_list.append(indiv_acc)
            bool_ys_list.append(ys.T)

            if args.wandb:
                wandb.log(
                    {
                        f"NCL optimum accuracy - {task_cfg['exp_name']}": indiv_acc,
                        f"NCL optimum loss - {task_cfg['exp_name']}": indiv_loss,
                    },
                    step=0,
                )

    # ── Mixed NCL computation (optional) ────────────────────────
    if args.wandb:
        if bool_ys_list:
            wandb.log(
                {"Mixed NCL optimum accuracy": get_optimal_ncl_acc(bool_ys_list)}, step=0
            )
        if cont_ys_list:
            mixed_loss = get_optimal_ncl_loss(cont_ys_list, cont_ncl_loss_list)
            wandb.log({"Mixed NCL optimum loss": mixed_loss}, step=0)
