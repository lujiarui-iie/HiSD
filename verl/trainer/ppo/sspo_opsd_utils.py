"""SSPO step-weighted advantage utilities for the existing multi-turn format."""

from collections import defaultdict
from typing import Sequence

import torch

METRIC_PREFIX = "sspo_opsd"


def _stat(values: torch.Tensor, percentile: float | None = None) -> float:
    values = values.detach().float()
    if values.numel() == 0:
        return 0.0
    if percentile is None:
        return values.mean().item()
    return torch.quantile(values, percentile / 100.0).item()


def _validate_inputs(
    advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    correct_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    beta: float,
    clip_eps: float,
) -> tuple[int, int]:
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "student and teacher log-probs must have identical shapes, got "
            f"{student_log_probs.shape} and {teacher_log_probs.shape}"
        )
    if student_log_probs.ndim != 2:
        raise ValueError("student and teacher log-probs must have shape [batch, response_length]")
    batch_size, response_length = student_log_probs.shape
    if response_mask.shape != (batch_size, response_length):
        raise ValueError(
            f"response_mask must have shape {(batch_size, response_length)}, got {response_mask.shape}"
        )
    if correct_mask.shape != (batch_size,):
        raise ValueError(f"correct_mask must have shape {(batch_size,)}, got {correct_mask.shape}")
    if advantages.ndim == 1:
        if advantages.shape != (batch_size,):
            raise ValueError(f"1D advantages must have shape {(batch_size,)}, got {advantages.shape}")
    elif advantages.shape != (batch_size, response_length):
        raise ValueError(
            "advantages must have shape [batch] or [batch, response_length], "
            f"got {advantages.shape}"
        )
    if len(traj_uids) != batch_size or len(turn_steps) != batch_size:
        raise ValueError("trajectory metadata must contain one value per batch item")
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    if not 0.0 <= clip_eps < 1.0:
        raise ValueError(f"clip_eps must be in [0, 1), got {clip_eps}")
    return batch_size, response_length


def compute_sspo_step_weights(
    advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    correct_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    *,
    beta: float = 1.0,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SSPO weights, grouping current-project rows by trajectory/turn.

    The returned row weights are one for correct trajectories and for invalid
    response positions. Only incorrect trajectories receive SSPO modulation.
    Duplicate rows for one ``(traj_uid, turn_step)`` share one support and one
    weight, so batch padding cannot advance a trajectory twice.
    """
    batch_size, response_length = _validate_inputs(
        advantages,
        student_log_probs,
        teacher_log_probs,
        response_mask,
        correct_mask,
        traj_uids,
        turn_steps,
        beta,
        clip_eps,
    )
    del response_length

    mask = response_mask.detach().bool()
    student = student_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    correct = correct_mask.detach().bool().to(device=student.device)
    # Keep control-flow checks on a small CPU list.  This avoids implicit
    # device synchronisation for every row when the batch lives on CUDA.
    correct_list = correct.detach().cpu().tolist()

    row_weights = torch.ones(batch_size, dtype=torch.float32, device=student.device)
    gap = teacher - student
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    trajectory_steps: dict[str, set[int]] = defaultdict(set)
    for index, (traj_uid, turn_step) in enumerate(zip(traj_uids, turn_steps)):
        key = (str(traj_uid), int(turn_step))
        groups[key].append(index)
        trajectory_steps[key[0]].add(key[1])

    supports: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    wrong_trajectory_count = 0
    for traj_uid, steps in trajectory_steps.items():
        ordered_steps = sorted(steps)
        wrong_indices = [
            index
            for step in ordered_steps
            for index in groups[(traj_uid, step)]
            if not correct_list[index]
        ]
        if not wrong_indices:
            continue
        wrong_trajectory_count += 1

        for turn_step in ordered_steps:
            indices = [
                index
                for index in groups[(traj_uid, turn_step)]
                if not correct_list[index]
            ]
            if not indices:
                continue
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=student.device)
            per_row_support = []
            for index in indices:
                valid_gap = gap[index][mask[index]]
                if valid_gap.numel() > 0:
                    per_row_support.append(valid_gap.mean())
            if not per_row_support:
                continue
            support = torch.stack(per_row_support).mean()

            signs = []
            for index in indices:
                valid_adv = advantages[index][mask[index]] if advantages.ndim == 2 else advantages[index : index + 1]
                sign = valid_adv.detach().float().mean().sign()
                signs.append(sign)
            sign = torch.stack(signs).mean().sign()
            if sign.item() == 0.0:
                sign = torch.tensor(-1.0, device=student.device)
            weight = torch.exp(sign * float(beta) * support).clamp_(1.0 - clip_eps, 1.0 + clip_eps)
            row_weights[index_tensor] = weight
            supports.append(support)
            weights.append(weight)

    support_tensor = torch.stack(supports) if supports else torch.empty(0, dtype=torch.float32, device=student.device)
    weight_tensor = torch.stack(weights) if weights else torch.empty(0, dtype=torch.float32, device=student.device)
    trajectory_count = torch.tensor(float(wrong_trajectory_count), dtype=torch.float32, device=student.device)
    return row_weights, support_tensor, weight_tensor, trajectory_count


def apply_sspo_advantage(
    advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    correct_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    *,
    beta: float = 1.0,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply SSPO's multiplicative step weights to GRPO advantages."""
    _, response_length = _validate_inputs(
        advantages,
        student_log_probs,
        teacher_log_probs,
        response_mask,
        correct_mask,
        traj_uids,
        turn_steps,
        beta,
        clip_eps,
    )
    row_weights, supports, weights, wrong_trajectory_count = compute_sspo_step_weights(
        advantages=advantages,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        correct_mask=correct_mask,
        traj_uids=traj_uids,
        turn_steps=turn_steps,
        beta=beta,
        clip_eps=clip_eps,
    )

    # Keep correct trajectories bitwise unchanged. The actor already applies
    # response_mask, while wrong trajectories receive the SSPO multiplier.
    total_advantages = advantages.clone()
    wrong = ~correct_mask.detach().bool().to(device=advantages.device)
    if advantages.ndim == 1:
        total_advantages[wrong] = total_advantages[wrong] * row_weights[wrong].to(total_advantages.dtype)
    else:
        total_advantages[wrong] = total_advantages[wrong] * row_weights[wrong].unsqueeze(-1).to(total_advantages.dtype)

    mask = response_mask.detach().bool()
    gap = (teacher_log_probs.detach().float() - student_log_probs.detach().float())[mask]
    metrics = {
        f"{METRIC_PREFIX}/beta": float(beta),
        f"{METRIC_PREFIX}/clip_eps": float(clip_eps),
        f"{METRIC_PREFIX}/correct_traj_count": float(correct_mask.detach().bool().sum().item()),
        f"{METRIC_PREFIX}/wrong_traj_count": float(wrong_trajectory_count.item()),
        f"{METRIC_PREFIX}/step_support_mean": _stat(supports),
        f"{METRIC_PREFIX}/step_support_p90": _stat(supports, 90.0),
        f"{METRIC_PREFIX}/step_weight_mean": _stat(weights),
        f"{METRIC_PREFIX}/step_weight_p10": _stat(weights, 10.0),
        f"{METRIC_PREFIX}/step_weight_p90": _stat(weights, 90.0),
        f"{METRIC_PREFIX}/teacher_student_gap_mean": _stat(gap),
        f"{METRIC_PREFIX}/response_length": float(mask.sum(dim=-1).float().mean().item()) if mask.numel() else 0.0,
    }
    del response_length
    return total_advantages, metrics
