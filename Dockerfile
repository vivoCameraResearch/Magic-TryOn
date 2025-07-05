FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

COPY . .

SHELL ["/bin/bash", "-c"]

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
RUN sudo dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt update
RUN apt install wget -y
RUN apt install git -y
RUN apt install unzip -y
RUN apt install cuda-toolkit-12-1 -y
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"

RUN source ~/miniconda3/bin/activate && \
    conda init --all && \
    conda create -n magictryon python==3.12.9 -y && \
    conda activate magictryon && \
    pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    conda install conda-content-trust -y && \
    conda install libmambapy -y && \
    conda install menuinst -y && \
    pip install -r requirements.txt && \
    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download LuckyLiGY/MagicTryOn --local-dir ./weights/MagicTryOn_14B_V1

RUN rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
    
RUN echo conda activate magictryon >> ~/.bashrc

# Customize TryOn
RUN wget https://github.com/dvorakchen/Magic-TryOn-Assets/releases/download/0.0.1/detail.pth -O weights/detail.pth
RUN wget https://github.com/dvorakchen/Magic-TryOn-Assets/releases/download/v0.0.2/gen_mask_ckpt.zip
RUN unzip gen_mask_ckpt.zip -d gen_mask_ckpt
RUN rm gen_mask_ckpt.zip
RUN cp gen_mask_ckpt/ckpt/ inference/customize/gen_mask/ckpt/ -r
RUN rm gen_mask_ckpt/ -rf

