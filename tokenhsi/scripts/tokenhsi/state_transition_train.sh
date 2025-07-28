#!/bin/bash

echo "开始训练四技能状态转移模型..."

# 确保必要的目录存在
mkdir -p tokenhsi/data/dataset_longterm_task_completion/task_plans/4_basic_skills_transition

# 方案1：使用修复后的配置文件直接训练
python ./tokenhsi/run.py --task HumanoidStateTransition4Skills \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/transition_skills/amp_humanoid_state_transition_4skills.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --num_envs 4096 \
    --headless

echo "状态转移模型训练完成！"