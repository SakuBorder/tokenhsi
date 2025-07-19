echo "开始训练四技能状态转移模型..."

python ./tokenhsi/run.py --task HumanoidStateTransition4Skills \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task.yaml \
    --cfg_env tokenhsi/data/cfg/transition_skills/amp_humanoid_state_transition_4skills.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --num_envs 4096 \
    --headless

echo "状态转移模型训练完成！"