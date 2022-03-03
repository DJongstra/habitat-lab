# Base image
FROM nvidia/cudagl:10.1-devel-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    libsm6 \
    libhdf5-dev \
    mesa-utils \
    xorg-dev \
    freeglut3-dev \
    pkg-config \
    wget \
    libbullet-dev \
    zip \
    unzip \
    net-tools \
    python-dev \
    zsh && \
    rm -rf /var/lib/apt/lists/*

# Install conda and dependencies.
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    export PATH=/opt/conda/bin:$PATH &&\
    conda install -q -y -c conda-forge ninja numpy pytest pytest-cov ccache hypothesis &&\
    pip install pip install pytest-parallel typeguard &&\
    conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Conda environment
RUN conda create -n habitat python=3.6 cmake=3.14.0

# Make RUN commands use the new habitat conda environment:
SHELL ["conda", "run", "-n", "habitat", "/bin/bash", "-c"]

# Install dependencies.
RUN conda install Cython pkgconfig h5py &&\
    conda install -c  conda-forge opencv -y &&\
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch &&\
    conda clean -ya

# Install habtiat-sim.
RUN conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Install for video
RUN conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge

# Prevent using cache if new commit found
ADD "https://api.github.com/repos/DJongstra/habitat-lab/commits/social_nav_v0?per_page=1" latest_commit

# Install habitat lab: social_nav branch
RUN cd / &&\
    git clone -b social_nav_v0 https://github.com/djongstra/habitat-lab.git


RUN cd /habitat-lab &&\
    pip install -r requirements.txt --progress-bar off &&\
    python setup.py develop --all
        

# Add base ddppo config(s)
ADD habitat_baselines/config/pointnav/ddppo_pointnav.yaml /habitat-lab/habitat_baselines/config/pointnav/

# Add social navigation configs
ADD configs/tasks/socialnav.yaml /habitat-lab/configs/tasks/
ADD habitat_baselines/config/pointnav/ddppo_pointnav_social.yaml /habitat-lab/habitat_baselines/config/pointnav/

# Add social navigation configs without the map/video options in the list
ADD configs/tasks/socialnav-nomap.yaml /habitat-lab/configs/tasks/
ADD habitat_baselines/config/pointnav/ddppo_pointnav_social-novid.yaml /habitat-lab/habitat_baselines/config/pointnav/

# Add experiment scripts.
ADD train.sh /
ADD evaluation.sh /

