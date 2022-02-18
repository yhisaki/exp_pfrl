FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib64
ENV MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco210


RUN apt-get update
RUN apt-get install -y \
  python3 \
  python3-pip \
  wget \
  git \
  patchelf \
  libosmesa6-dev

RUN mkdir -p /opt/.mujoco &&\
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz &&\
  tar -xf mujoco.tar.gz -C /opt/.mujoco &&\
  rm mujoco.tar.gz

RUN pip install "mujoco-py<2.2,>=2.1" &&\
  python3 -c "import mujoco_py" &&\
  chmod -R 777 /usr/local/lib/python3.8/dist-packages/mujoco_py

ADD requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt