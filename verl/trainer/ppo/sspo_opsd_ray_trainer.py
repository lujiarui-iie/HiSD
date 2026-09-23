"""SSPO trainer using the existing privileged OPSD teacher and trajectory rows."""


import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer
from verl.trainer.ppo.sspo_opsd_utils import apply_sspo_advantage


class SSPOOPSDRayTrainer(SkillSDRayTrainer):
    """Apply SSPO's multiplicative step weights after standard GRPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sspo_cfg = self.config.algorithm.get("sspo_opsd", {})
        if not sspo_cfg.get("enable", True):
            raise ValueError("algorithm.sspo_opsd.enable must be true for SSPO-OPSD training")
        self.sspo_beta = float(sspo_cfg.get("beta", 1.0))
        self.sspo_clip_eps = float(sspo_cfg.get("clip_eps", 0.2))
        self.sspo_correct_reward_threshold = float(sspo_cfg.get("correct_reward_threshold", 1.0))
        self._sspo_teacher_metrics = {}
        if self.sspo_beta < 0:
            raise ValueError("algorithm.sspo_opsd.beta must be non-negative")
        if not 0.0 <= self.sspo_clip_eps < 1.0:
            raise ValueError("algorithm.sspo_opsd.clip_eps must be in [0, 1)")

    def _correct_mask(self, batch: DataProto) -> torch.Tensor:
        rewards = batch.non_tensor_batch.get("episode_rewards")
        if rewards is not None:
            rewards = np.asarray(rewards)
            return torch.as_tensor(
                rewards.astype(np.float64) >= self.sspo_correct_reward_threshold,
                dtype=torch.bool,
                device=batch.batch["old_log_probs"].device,
            )
        if "token_level_scores" not in batch.batch:
            raise KeyError("SSPO-OPSD requires episode_rewards metadata or token_level_scores")
        return (
            batch.batch["token_level_scores"].detach().float().sum(dim=-1)
            >= self.sspo_correct_reward_threshold
        )

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """Run the privileged teacher only for incorrect, trainable rows.

        SSPO uses the teacher-student gap to reweight failed trajectories.  A
        correct trajectory therefore needs no teacher forward at all; its
        teacher log-probabilities are represented by the detached student
        values, yielding a zero gap while avoiding unnecessary computation.
        """
        correct_mask = self._correct_mask(batch)
        student_log_probs = batch.batch["old_log_probs"]
        teacher_log_probs = student_log_probs.detach().clone()

        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            raise KeyError("SSPO-OPSD requires response_mask before teacher computation")
        trainable_mask = response_mask.detach().bool().any(dim=-1)
        wrong_mask = (~correct_mask) & trainable_mask
        wrong_indices = torch.nonzero(wrong_mask, as_tuple=False).flatten()

        self._sspo_teacher_metrics = {
            "sspo_opsd/teacher_forward_count": float(wrong_indices.numel()),
            "sspo_opsd/teacher_forward_frac": float(wrong_indices.numel()) / max(len(batch), 1),
        }
        if wrong_indices.numel() == 0:
            return teacher_log_probs

        teacher_batch = build_teacher_batch(
            batch=batch.select_idxs(wrong_indices.detach().cpu().numpy()),
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )
        # The selected error rows are generally not divisible by the actor
        # data-parallel world size.  Pad only this temporary teacher batch for
        # dispatch, then remove the repeated rows before restoring positions.
        teacher_divisor = max(int(getattr(self.actor_rollout_wg, "world_size", 1)), 1)
        teacher_batch, teacher_pad_size = pad_dataproto_to_divisor(
            teacher_batch,
            teacher_divisor,
        )
        self._sspo_teacher_metrics.update(
            {
                "sspo_opsd/teacher_forward_pad_size": float(teacher_pad_size),
                "sspo_opsd/teacher_forward_divisor": float(teacher_divisor),
            }
        )
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_output = unpad_dataproto(teacher_output, pad_size=teacher_pad_size)
        selected_teacher_log_probs = teacher_output.batch["old_log_probs"]
        device_indices = wrong_indices.to(device=teacher_log_probs.device)
        teacher_log_probs[device_indices] = selected_teacher_log_probs.to(
            device=teacher_log_probs.device,
            dtype=teacher_log_probs.dtype,
        )
        return teacher_log_probs

    def _postprocess_advantages(self, batch: DataProto, metrics: dict) -> DataProto:
        required = {"advantages", "old_log_probs", "teacher_log_probs", "response_mask"}
        missing = required.difference(batch.batch.keys())
        if missing:
            raise KeyError(f"SSPO-OPSD batch is missing tensor fields: {sorted(missing)}")
        if "traj_uid" not in batch.non_tensor_batch or "turn_step" not in batch.non_tensor_batch:
            raise KeyError("SSPO-OPSD requires traj_uid and turn_step metadata")

        correct_mask = self._correct_mask(batch)
        advantages, sspo_metrics = apply_sspo_advantage(
            advantages=batch.batch["advantages"],
            student_log_probs=batch.batch["old_log_probs"],
            teacher_log_probs=batch.batch["teacher_log_probs"],
            response_mask=batch.batch["response_mask"],
            correct_mask=correct_mask,
            traj_uids=batch.non_tensor_batch["traj_uid"],
            turn_steps=batch.non_tensor_batch["turn_step"],
            beta=self.sspo_beta,
            clip_eps=self.sspo_clip_eps,
        )
        batch.batch["advantages"] = advantages
        # SSPO has no value-function update in the GRPO path. Keep the
        # diagnostic ``returns`` aligned with the reweighted advantages, as
        # in the reference SSPO trainer.
        if "returns" in batch.batch:
            batch.batch["returns"] = advantages.clone()
        metrics.update(getattr(self, "_sspo_teacher_metrics", {}))
        metrics.update(sspo_metrics)
        return batch
