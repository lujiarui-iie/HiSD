"""HiSD top-K supervision utilities.

The detached adaptation metadata is computed before actor optimization, while
the final forward KL is recomputed from the current actor logits with gradient.
"""

from collections import defaultdict
from typing import Sequence

import torch


METRIC_PREFIX = "hisd_task"


def _tensor_stat(values: torch.Tensor, stat: str) -> float:
    values = values.detach().float()
    if values.numel() == 0:
        return 0.0
    if stat == "mean":
        return values.mean().item()
    if stat == "max":
        return values.max().item()
    if stat.startswith("p"):
        return torch.quantile(values, float(stat[1:]) / 100.0).item()
    raise ValueError(f"unsupported statistic: {stat}")


def _distribution_metrics(name: str, values: torch.Tensor, stats: Sequence[str]) -> dict[str, float]:
    return {
        f"{METRIC_PREFIX}/adaptation/{name}_{stat}": _tensor_stat(values, stat)
        for stat in stats
    }


def build_hisd_metadata(
    teacher_topk_logits: torch.Tensor,
    student_topk_logits: torch.Tensor,
    response_mask: torch.Tensor,
    traj_uids: Sequence[object],
    turn_steps: Sequence[object],
    *,
    lambda_base: float,
    gamma: float,
    alpha: float,
    min_support_size: int = 2,
    random_k: int = 8,
    topk_source: str = "teacher",
    support_selector: str = "dynamic_top_p",
    coefficient_mode: str = "prefix_suffix_sigmoid",
    kappa: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Build detached token supports and per-turn OPSD weights.

    Duplicate copies of the same ``(traj_uid, turn_step)`` can be introduced by
    batch padding. They share one turn divergence and advance the trajectory
    prefix exactly once.
    """
    if teacher_topk_logits.shape != student_topk_logits.shape:
        raise ValueError(
            "teacher and student Top-K logits must have identical shapes, got "
            f"{teacher_topk_logits.shape} and {student_topk_logits.shape}"
        )
    if teacher_topk_logits.ndim != 3:
        raise ValueError("Top-K logits must have shape [batch, response_length, K]")

    batch_size, response_length, topk = teacher_topk_logits.shape
    if response_mask.shape != (batch_size, response_length):
        raise ValueError(
            f"response_mask must have shape {(batch_size, response_length)}, got {response_mask.shape}"
        )
    if len(traj_uids) != batch_size or len(turn_steps) != batch_size:
        raise ValueError("trajectory metadata must contain one value per batch item")
    if not 1 <= min_support_size <= topk:
        raise ValueError(f"min_support_size must be in [1, {topk}], got {min_support_size}")
    if topk_source not in {"teacher", "student"}:
        raise ValueError(
            f"topk_source must be 'teacher' or 'student', got {topk_source!r}"
        )
    if support_selector not in {"dynamic_top_p", "random"}:
        raise ValueError(
            "support_selector must be 'dynamic_top_p' or 'random', "
            f"got {support_selector!r}"
        )
    if support_selector == "random" and not 1 <= random_k <= topk:
        raise ValueError(f"random_k must be in [1, {topk}], got {random_k}")
    prefix_suffix_modes = {"prefix_suffix_sigmoid", "only_prefix", "only_suffix"}
    if coefficient_mode not in {"position", *prefix_suffix_modes}:
        raise ValueError(
            "coefficient_mode must be 'prefix_suffix_sigmoid', 'only_prefix', "
            "'only_suffix', or 'position', "
            f"got {coefficient_mode!r}"
        )
    if coefficient_mode in prefix_suffix_modes and kappa <= 0:
        raise ValueError(
            f"kappa must be positive for {coefficient_mode}, got {kappa}"
        )
    if lambda_base < 0 or not 0.0 <= gamma <= 1.0 or alpha < 0:
        raise ValueError("HiSD coefficients must be non-negative and gamma must be in [0, 1]")

    teacher_logits = teacher_topk_logits.detach().float()
    student_logits = student_topk_logits.detach().float()
    mask = response_mask.detach().bool()

    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_probs = student_log_probs.exp()

    token_delta = torch.sum(
        student_probs * (student_log_probs - teacher_log_probs), dim=-1
    )
    token_delta = token_delta * mask

    rho = torch.exp(-alpha * token_delta)
    if support_selector == "dynamic_top_p":
        support_probs = teacher_probs if topk_source == "teacher" else student_probs
        cumulative_support_prob = support_probs.cumsum(dim=-1)
        # Include the first token whose addition reaches rho.
        support_mask = (cumulative_support_prob - support_probs) < rho.unsqueeze(-1)
        support_mask[..., :min_support_size] = True
    else:
        random_ids = torch.rand_like(teacher_logits).topk(random_k, dim=-1).indices
        support_mask = torch.zeros_like(teacher_logits, dtype=torch.bool)
        support_mask.scatter_(-1, random_ids, True)
    support_mask &= mask.unsqueeze(-1)

    token_counts = mask.sum(dim=-1).clamp(min=1)
    turn_delta = token_delta.sum(dim=-1) / token_counts

    turn_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    trajectory_turns: dict[str, set[int]] = defaultdict(set)
    for index, (traj_uid, turn_step) in enumerate(zip(traj_uids, turn_steps)):
        key = (str(traj_uid), int(turn_step))
        turn_to_indices[key].append(index)
        trajectory_turns[key[0]].add(key[1])

    turn_weights = torch.empty(batch_size, dtype=torch.float32, device=teacher_logits.device)
    prefix_deltas = torch.empty(batch_size, dtype=torch.float32, device=teacher_logits.device)
    suffix_deltas = (
        torch.empty(batch_size, dtype=torch.float32, device=teacher_logits.device)
        if coefficient_mode in prefix_suffix_modes
        else None
    )
    coefficients = torch.empty_like(turn_weights) if suffix_deltas is not None else None
    for traj_uid, steps in trajectory_turns.items():
        ordered_steps = sorted(steps)
        step_data = []
        for turn_step in ordered_steps:
            indices = turn_to_indices[(traj_uid, turn_step)]
            current_delta = turn_delta[indices].mean()
            step_data.append((indices, current_delta))

        prefix_delta = torch.zeros((), dtype=torch.float32, device=teacher_logits.device)
        if coefficient_mode == "position":
            num_steps = len(step_data)
            for step_index, (indices, current_delta) in enumerate(step_data, start=1):
                weight = lambda_base * (num_steps + 1 - step_index) / num_steps
                turn_weights[indices] = weight
                prefix_deltas[indices] = prefix_delta
                prefix_delta = gamma * prefix_delta + current_delta
        else:
            assert suffix_deltas is not None and coefficients is not None
            suffix_by_step = []
            suffix_delta = torch.zeros((), dtype=torch.float32, device=teacher_logits.device)
            for _, current_delta in reversed(step_data):
                suffix_delta = current_delta + gamma * suffix_delta
                suffix_by_step.append(suffix_delta)
            suffix_by_step.reverse()

            for (indices, current_delta), suffix_delta in zip(step_data, suffix_by_step):
                if coefficient_mode == "prefix_suffix_sigmoid":
                    coefficient_logit = suffix_delta - prefix_delta
                elif coefficient_mode == "only_prefix":
                    coefficient_logit = -prefix_delta
                elif coefficient_mode == "only_suffix":
                    coefficient_logit = suffix_delta
                coefficient = torch.sigmoid(coefficient_logit / kappa)
                turn_weights[indices] = lambda_base * coefficient
                prefix_deltas[indices] = prefix_delta
                suffix_deltas[indices] = suffix_delta
                coefficients[indices] = coefficient
                prefix_delta = gamma * prefix_delta + current_delta

    valid_support_sizes = support_mask.sum(dim=-1)[mask]
    valid_delta = token_delta[mask]
    valid_rho = rho[mask]

    metrics = {}
    metrics.update(_distribution_metrics("reverse_kl", valid_delta, ("mean", "p90")))
    metrics.update(_distribution_metrics("dynamic_top_p", valid_rho, ("mean", "p10")))
    metrics.update(
        _distribution_metrics(
            "support_size", valid_support_sizes.float(), ("mean", "p90")
        )
    )
    metrics.update(_distribution_metrics("prefix_D", prefix_deltas, ("mean", "p90")))
    metrics.update(_distribution_metrics("lambda", turn_weights, ("mean", "p10")))
    if suffix_deltas is not None:
        metrics.update(
            _distribution_metrics("suffix_D", suffix_deltas, ("mean", "p90"))
        )
        metrics.update(
            _distribution_metrics(
                "suffix_minus_prefix_D",
                suffix_deltas - prefix_deltas,
                ("mean", "p10", "p90"),
            )
        )
        metrics.update(
            _distribution_metrics(
                "coefficient", coefficients, ("mean", "p10", "p90")
            )
        )
    metrics[f"{METRIC_PREFIX}/adaptation/support_singleton_ratio"] = (
        (valid_support_sizes == 1).float().mean().item()
        if valid_support_sizes.numel()
        else 0.0
    )

    tensors = {
        "hisd_support_mask": support_mask,
        "hisd_turn_weight": turn_weights,
    }
    return tensors, metrics


def compute_hisd_loss(
    student_topk_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    support_mask: torch.Tensor,
    response_mask: torch.Tensor,
    turn_weight: torch.Tensor,
    renormalize: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute action-normalized distillation loss on the detached dynamic support."""
    if student_topk_logits.shape != teacher_topk_logits.shape:
        raise ValueError("student and teacher Top-K logits must have identical shapes")
    if support_mask.shape != student_topk_logits.shape:
        raise ValueError("support_mask must match the Top-K logits shape")

    response_mask = response_mask.bool()
    support_mask = support_mask.bool() & response_mask.unsqueeze(-1)
    teacher_logits = teacher_topk_logits.detach().float()
    student_logits = student_topk_logits.float()

    if renormalize:
        floor = torch.finfo(student_logits.dtype).min
        teacher_logits = teacher_logits.masked_fill(~support_mask, floor)
        student_logits = student_logits.masked_fill(~support_mask, floor)

    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    token_terms = teacher_probs * (teacher_log_probs - student_log_probs)
    if not renormalize:
        token_terms = token_terms.masked_fill(~support_mask, 0.0)
    token_kl = torch.sum(token_terms, dim=-1)
    token_kl = token_kl * response_mask

    token_counts = response_mask.sum(dim=-1).clamp(min=1)
    action_loss = token_kl.sum(dim=-1) / token_counts
    loss = (action_loss * turn_weight.float()).mean()

    with torch.no_grad():
        valid_action_kl = action_loss[response_mask.any(dim=-1)]
        metrics = {
            f"{METRIC_PREFIX}/adaptation/loss": loss.detach().item(),
            f"{METRIC_PREFIX}/adaptation/train_forward_kl_action_mean": _tensor_stat(valid_action_kl, "mean"),
            f"{METRIC_PREFIX}/adaptation/train_forward_kl_action_p90": _tensor_stat(valid_action_kl, "p90"),
        }
    return loss, metrics
