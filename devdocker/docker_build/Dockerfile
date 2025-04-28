FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables to non-interactive (This prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Update and install packages
RUN apt update && apt install -y \
    sudo \
    lsb-release \
    software-properties-common \
    curl \
    wget \
    vim \
    htop \
    git \
    libgl1-mesa-glx

COPY install_conda.sh /tools/

RUN bash /tools/install_conda.sh

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

# Create a new group and user with the passed user name, then add them to the 'sudo' group
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${USER_NAME} -s /bin/bash ${USER_NAME} && \
    mkdir -p /etc/sudoers.d/ && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER_NAME}

# Change to non-root privilege
USER ${USER_NAME}

WORKDIR /home/${USER_NAME}

RUN /miniconda3/bin/conda create -n "land" -y python=3.12 cmake=3.31.2
SHELL ["/miniconda3/bin/conda", "run", "--no-capture-output", "-n", "land", "/bin/bash", "-c"]
RUN conda install -y numpy pytorch torchvision pytorch-cuda=12.4 ffmpeg -c pytorch -c nvidia

RUN pip install plyfile tqdm open3d mediapy lpips trimesh scikit-image scikit-learn \
    opencv-python pynvml matplotlib pandas psutil tensorboard einops vtk roma hydra-core "imageio[ffmpeg]" \
    stable-baselines3 "gymnasium[other,classic-control]"
RUN conda install conda-forge::cgal
RUN conda install tbb-devel

RUN echo "alias conda_on='source /miniconda3/etc/profile.d/conda.sh'" >> ~/.bashrc

# Set the default command for the container
CMD ["bash"]