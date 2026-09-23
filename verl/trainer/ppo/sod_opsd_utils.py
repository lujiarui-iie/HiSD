"""Utilities for SOD-style step-weighted on-policy self-distillation."""

from collections import defaultdict
from typing import Sequence

import torch

METRIC_PREFIX = "sod_opsd"


def _tensor_stat(values: torch.Tensor, stat: str) -> float:
    values = values.detach().float()
    if values.numel() == 0:
        return 0.0
    if stat == "mean":
        return values.mean().item()
    if stat.startswith("p"):
        return torch.quantile(values, float(stat[1:]) / 100.0).item()
    raise ValueError(f"unsupported statistic: {stat}")


def _validate_inputs(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    *,
    epsilon: float,
    delta: float,
) -> tuple[int, int]:
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(f"student and teacher log-probs must have identical shapes, got {student_log_probs.shape} and {teacher_log_probs.shape}")
    if student_log_probs.ndim != 2:
        raise ValueError("student and teacher log-probs must have shape [batch, response_length]")

    batch_size, response_length = student_log_probs.shape
    if response_mask.shape != (batch_size, response_length):
        raise ValueError(f"response_mask must have shape {(batch_size, response_length)}, got {response_mask.shape}")
    if len(traj_uids) != batch_size or len(turn_steps) != batch_size:
        raise ValueError("trajectory metadata must contain one value per batch item")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    return batch_size, response_length


def compute_sod_step_weights(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    *,
    epsilon: float = 1e-6,
    delta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SOD weights using this project's trajectory/turn metadata.

    Each batch item represents one agent turn. Copies of the same
    ``(traj_uid, turn_step)`` introduced by batch padding share one divergence
    and one weight, and advance the trajectory only once.

    Returns:
        Per-item weights, per-unique-turn divergences, and per-unique-turn
        weights. Unique-turn tensors are intended for unbiased logging.
    """
    batch_size, _ = _validate_inputs(
        student_log_probs,
        teacher_log_probs,
        response_mask,
        traj_uids,
        turn_steps,
        epsilon=epsilon,
        delta=delta,
    )

    student = student_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    mask = response_mask.detach().bool()
    token_counts = mask.sum(dim=-1).clamp_min(1)
    absolute_gap = torch.where(mask, (student - teacher).abs(), 0.0)
    item_divergences = absolute_gap.sum(dim=-1) / token_counts

    turn_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    trajectory_turns: dict[str, set[int]] = defaultdict(set)
    for index, (traj_uid, turn_step) in enumerate(zip(traj_uids, turn_steps)):
        key = (str(traj_uid), int(turn_step))
        turn_to_indices[key].append(index)
        trajectory_turns[key[0]].add(key[1])

    item_weights = torch.zeros(batch_size, dtype=torch.float32, device=student.device)
    unique_divergences = []
    unique_weights = []
    max_weight = 1.0 + delta

    for traj_uid, steps in trajectory_turns.items():
        cumulative_ratio = torch.ones((), dtype=torch.float32, device=student.device)
        previous_divergence = None
        for step_index, turn_step in enumerate(sorted(steps)):
            indices = turn_to_indices[(traj_uid, turn_step)]
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=student.device)
            divergence = item_divergences[index_tensor].mean()

            if step_index == 0:
                weight = torch.ones((), dtype=torch.float32, device=student.device)
            else:
                ratio = (previous_divergence + epsilon) / (divergence + epsilon)
                cumulative_ratio = cumulative_ratio * ratio
                weight = torch.clamp(cumulative_ratio, max=max_weight)

            item_weights[index_tensor] = weight
            unique_divergences.append(divergence)
            unique_weights.append(weight)
            previous_divergence = divergence

    if unique_divergences:
        divergence_tensor = torch.stack(unique_divergences)
        weight_tensor = torch.stack(unique_weights)
    else:
        divergence_tensor = torch.empty(0, dtype=torch.float32, device=student.device)
        weight_tensor = torch.empty(0, dtype=torch.float32, device=student.device)
    return item_weights, divergence_tensor, weight_tensor


def apply_sod_opsd_advantage(
    advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    *,
    epsilon: float = 1e-6,
    delta: float = 0.5,
    opd_coef: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Add the SOD-weighted OPSD signal to an existing GRPO advantage."""
    batch_size, response_length = _validate_inputs(
        student_log_probs,
        teacher_log_probs,
        response_mask,
        traj_uids,
        turn_steps,
        epsilon=epsilon,
        delta=delta,
    )
    if opd_coef < 0:
        raise ValueError(f"opd_coef must be non-negative, got {opd_coef}")
    if advantages.ndim == 1:
        if advantages.shape[0] != batch_size:
            raise ValueError(f"1D advantages must have length {batch_size}, got {advantages.shape}")
    elif advantages.shape != (batch_size, response_length):
        raise ValueError(f"advantages must have shape [batch] or [batch, response_length], got {advantages.shape}")

    item_weights, turn_divergences, turn_weights = compute_sod_step_weights(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        traj_uids=traj_uids,
        turn_steps=turn_steps,
        epsilon=epsilon,
        delta=delta,
    )

    mask = response_mask.detach().bool()
    student = student_log_probs.detach().float()
    teacher = teacher_log_probs.detach().float()
    local_advantage = torch.where(mask, teacher - student, 0.0)
    weighted_opd = opd_coef * item_weights.unsqueeze(-1) * local_advantage

    if opd_coef == 0:
        total_advantage = advantages
    else:
        base_advantage = advantages.detach().float()
        if base_advantage.ndim == 1:
            base_advantage = base_advantage.unsqueeze(-1)
        total_advantage = base_advantage + weighted_opd

    valid_local_advantage = local_advantage[mask]
    valid_weighted_opd = weighted_opd[mask]
    metrics = {
        f"{METRIC_PREFIX}/divergence_mean": _tensor_stat(turn_divergences, "mean"),
        f"{METRIC_PREFIX}/divergence_p90": _tensor_stat(turn_divergences, "p90"),
        f"{METRIC_PREFIX}/step_weight_mean": _tensor_stat(turn_weights, "mean"),
        f"{METRIC_PREFIX}/step_weight_p10": _tensor_stat(turn_weights, "p10"),
        f"{METRIC_PREFIX}/step_weight_p90": _tensor_stat(turn_weights, "p90"),
        f"{METRIC_PREFIX}/local_adv_mean": _tensor_stat(valid_local_advantage, "mean"),
        f"{METRIC_PREFIX}/weighted_opd_mean": _tensor_stat(valid_weighted_opd, "mean"),
        f"{METRIC_PREFIX}/opd_coef": float(opd_coef),
    }
    return total_advantage, metrics
