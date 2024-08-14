FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git \
    git-lfs \
    wget \
    curl \
    zip && \
    set -xe &&\
    apt-get install -y python3-pip
    
RUN apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

RUN  apt-get dist-upgrade -y && \
     apt-get -y autoremove && \
     apt-get clean
     
RUN pip install --upgrade pip
RUN apt-get install libjpeg-dev libpng-dev libtiff-dev -y
RUN git-lfs install
RUN apt install vim -y
RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt requirements.txt
RUN pip install packaging==24.1
RUN pip install -r requirements.txt


WORKDIR /app

COPY background_removal/ /app/background_removal
COPY checkpoints/ /app/checkpoints
COPY configs/ /app/configs
COPY deepfillv2/ /app/deepfillv2
COPY model_weights/ /app/model_weights
COPY utils/ /app/utils
COPY app.py /app/app.py
COPY inpaint.py /app/inpaint.py
COPY mascot.png /app/mascot.png
COPY README.md /app/README.md
COPY Dockerfile /app/Dockerfile

CMD ["python3","-u", "app.py"]
