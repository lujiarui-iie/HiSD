"""SOD-OPSD trainer built on the existing privileged self-teacher pipeline."""

from verl import DataProto
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer
from verl.trainer.ppo.sod_opsd_utils import apply_sod_opsd_advantage


class SODOPSDRayTrainer(SkillSDRayTrainer):
    """Inject SOD's step-weighted sampled-token OPSD signal into GRPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sod_cfg = self.config.algorithm.get("sod_opsd", {})
        if not sod_cfg.get("enable", True):
            raise ValueError("algorithm.sod_opsd.enable must be true for SOD-OPSD training")
        self.sod_epsilon = float(sod_cfg.get("epsilon", 1e-6))
        self.sod_delta = float(sod_cfg.get("delta", 0.5))
        self.sod_opd_coef = float(sod_cfg.get("opd_coef", 1.0))

    def _postprocess_advantages(self, batch: DataProto, metrics: dict) -> DataProto:
        required_batch_keys = {
            "advantages",
            "old_log_probs",
            "teacher_log_probs",
            "response_mask",
        }
        missing_batch_keys = required_batch_keys.difference(batch.batch.keys())
        if missing_batch_keys:
            raise KeyError(f"SOD-OPSD batch is missing tensor fields: {sorted(missing_batch_keys)}")

        required_metadata = {"traj_uid", "turn_step"}
        missing_metadata = required_metadata.difference(batch.non_tensor_batch.keys())
        if missing_metadata:
            raise KeyError(f"SOD-OPSD batch is missing trajectory fields: {sorted(missing_metadata)}")

        advantages, sod_metrics = apply_sod_opsd_advantage(
            advantages=batch.batch["advantages"],
            student_log_probs=batch.batch["old_log_probs"],
            teacher_log_probs=batch.batch["teacher_log_probs"],
            response_mask=batch.batch["response_mask"],
            traj_uids=batch.non_tensor_batch["traj_uid"],
            turn_steps=batch.non_tensor_batch["turn_step"],
            epsilon=self.sod_epsilon,
            delta=self.sod_delta,
            opd_coef=self.sod_opd_coef,
        )
        batch.batch["advantages"] = advantages
        metrics.update(sod_metrics)
        return batch
