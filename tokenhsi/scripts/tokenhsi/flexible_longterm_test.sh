echo "测试灵活长技能组合..."

python ./tokenhsi/run.py --task HumanoidFlexibleLongTerm \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_transformer_multi_task_longterm.yaml \
    --cfg_env tokenhsi/data/cfg/longterm_task_completion/amp_humanoid_flexible_longterm.yaml \
    --motion_file tokenhsi/data/dataset_loco_sit_carry_climb.yaml \
    --hrl_checkpoint output/tokenhsi/ckpt_stage1.pth \
    --checkpoint output/tokenhsi/ckpt_state_transition.pth \
    --test \
    --num_envs 16

echo "灵活长技能测试完成！"