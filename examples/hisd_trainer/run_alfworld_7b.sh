set -x
ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# HiSD hyperparameters
lambda_base=0.01
coefficient_mode=prefix_suffix_sigmoid
kappa=1.0
gamma=0.9
alpha=3.0
support_selector=dynamic_top_p
topk=256
topk_source=teacher
min_support_size=1
renormalize=true
skill_all=false

train_data_size=16
val_data_size=128
group_size=8
renormalize_suffix=""
topk_source_suffix=""
coefficient_mode_suffix=""
if [[ "$coefficient_mode" == "prefix_suffix_sigmoid" ]]; then
    coefficient_mode_suffix="_prefixsuffix"
elif [[ "$coefficient_mode" == "only_prefix" ]]; then
    coefficient_mode_suffix="_onlyprefix"
elif [[ "$coefficient_mode" == "only_suffix" ]]; then
    coefficient_mode_suffix="_onlysuffix"
fi
if [[ "$renormalize" == "false" ]]; then
    renormalize_suffix="_norenormalize"
fi
if [[ "$topk_source" == "student" ]]; then
    topk_source_suffix="_studenttopk"
fi
experiment_name="hisd_qwen2.5_7b_k${topk}_lambda${lambda_base}_gamma${gamma}_alpha${alpha}${renormalize_suffix}${topk_source_suffix}${coefficient_mode_suffix}"
export ALFWORLD_DATA=$HOME/data/alfworld

export WANDB_API_KEY=your_key_here

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_hisd \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    +actor_rollout_ref.actor.hisd_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.hisd.enable=True \
    +algorithm.hisd.lambda_base=$lambda_base \
    +algorithm.hisd.coefficient_mode=$coefficient_mode \
    +algorithm.hisd.kappa=$kappa \
    +algorithm.hisd.gamma=$gamma \
    +algorithm.hisd.alpha=$alpha \
    +algorithm.hisd.support_selector=$support_selector \
    +algorithm.hisd.topk=$topk \
    +algorithm.hisd.topk_source=$topk_source \
    +algorithm.hisd.min_support_size=$min_support_size \
    +algorithm.hisd.renormalize=$renormalize \
    +algorithm.hisd.skills_dir=skills/alfworld \
    +algorithm.hisd.skill_all=$skill_all \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='hisd_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
