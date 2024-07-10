FROM osrf/ros:noetic-desktop-full
# 'desktop-full' includes gazebo9

# NVIDIA env variables. No longer required
# ENV NVIDIA_VISIBLE_DEVICES \
#     ${NVIDIA_VISIBLE_DEVICES:-all}
# ENV NVIDIA_DRIVER_CAPABILITIES \
#     ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
# Ensure bin/sh (default shell) is set to bash
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
### Fix entrypoint.sh file
RUN sed -i 's/set -e//' /ros_entrypoint.sh

# Install basic packages
RUN apt update -y \
	&& apt-get install -y wget nano git \
	&& rm -rf /var/lib/apt/lists/*

# Install ROS packages
RUN apt update -y \
	&& apt-get install -y ros-noetic-ros-numpy ros-noetic-tf* python3-pykdl \
	&& rm -rf /var/lib/apt/lists/*

# Install conda and modify entrypoint to init conda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
	&& ln -s /root/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
	&& echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
	&& echo "conda activate base" >> ~/.bashrc
RUN conda --version

# Ensure image runs bash
CMD ["bash"]