#!/bin/bash

python ./tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_g1_humanoid_traj.yaml  \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco_g1.yaml \
    --checkpoint /home/dy/dy/code/tokenhsi/output/Humanoid_07-23-08-27/nn/Humanoid.pth \
    --test \
    --num_envs 16 