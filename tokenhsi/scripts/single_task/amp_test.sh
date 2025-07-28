#!/bin/bash

python ./tokenhsi/run.py --task HumanoidAMP \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_humanoid_traj.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco.yaml \
    --checkpoint /home/dy/dy/code/tokenhsi/output/Humanoid_21-17-16-46/nn/Humanoid.pth\
    --test \
    --num_envs 16