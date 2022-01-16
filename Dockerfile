FROM registry.gitlab.com/uantwerp/student/researchproject2021/office-simulator/embodied-docker:20210318-notebook

# Wandb for custom logging in the cloud.
RUN pip install wandb

# Remove default habitat-lab from base image.
RUN rm -r /habitat-lab

RUN cd / &&\
    git clone -b social_nav_v0 https://github.com/djongstra/habitat-lab.git

RUN cd /habitat-lab &&\
    pip install -r requirements.txt --progress-bar off &&\
    python setup.py develop --all

# Add experiment scripts.
ADD train.sh /
ADD evaluation.sh /

# Social navigation debug environment
ADD configs/tasks/room3x3.yaml /habitat-lab/configs/tasks/
ADD habitat_baselines/config/pointnav/ddppo_pointnav_social.yaml /habitat-lab/habitat_baselines/config/pointnav/

