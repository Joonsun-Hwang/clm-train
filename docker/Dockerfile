# Builds GPU docker image of PyTorch
# Uses multi-staged approach to reduce size

# Environment Variables
ARG PYTHON_VERSION
ARG CUDA_VERSION
ARG WHEEL_VERSION
ARG CUDNN_VERSION
ARG UBUNTU_VERSION

# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS compile-image

# Specify py version
# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Create our conda env
RUN conda create --name accelerate python=${PYTHON_VERSION} ipython jupyter pip
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/accelerate/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Activate the conda env and install torch + accelerate
RUN source activate accelerate && \
    python3 -m pip install --no-cache-dir \
    git+https://github.com/huggingface/accelerate#egg=accelerate[testing,test_trackers] \
    --extra-index-url https://download.pytorch.org/whl/cu${WHEEL_VERSION}
# Install additional libraries if you want
RUN python3 -m pip install --no-cache-dir \
    peft adapter-transformers bert-score gpustat nltk nvidia-ml-py3 wandb

# Stage 2
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build-image
# FROM nvcr.io/nvidia/pytorch:23.01-py3
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Activate the virtualenv
RUN echo "source activate accelerate" >> ~/.bashrc
CMD ["/bin/bash"]