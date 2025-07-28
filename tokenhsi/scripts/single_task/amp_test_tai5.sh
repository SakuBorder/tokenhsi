#!/bin/bash

python ./tokenhsi/run.py --task HumanoidAMP \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task.yaml \
    --cfg_env tokenhsi/data/cfg/basic_interaction_skills/amp_tai5_traj.yaml \
    --motion_file tokenhsi/data/dataset_amass_loco/dataset_amass_loco_tai5.yaml \
    --checkpoint /home/dy/dy/code/tokenhsi/output/Humanoid_27-23-11-51/nn/Humanoid.pth\
    --test \
    --num_envs 16