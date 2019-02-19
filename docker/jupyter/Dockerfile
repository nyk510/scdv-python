FROM ubuntu:16.04
# All-In-One Dockerfile
# support cpu 
# If you use gpu, see https://github.com/nyk510/docker-zoo/blob/master/machine-learning/docker/gpu/Dockerfile

ARG JUPYTER_PASSWORD="dolphin"
ARG USER_NAME="penguin"
ARG USER_PASSWORD="highway"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install dependences
RUN apt-get update --fix-missing && \
  apt-get install -y \
    wget \
    bzip2 \ 
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion \
    sudo \
    git \
    zsh \
    openssh-server \
    wget \
    gcc \
    g++ \
    libatlas-base-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    curl \
    make \
    unzip \
    # MeCab
    swig mecab libmecab-dev mecab-ipadic-utf8 \
    cmake --fix-missing

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Install miniconda dir and add path
ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:${PATH}

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.1.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh

RUN conda install -y conda

RUN conda install -y \
  numpy \
  scipy \
  scikit-learn

# install lightGBM
RUN git clone --recursive https://github.com/Microsoft/LightGBM && \
  cd LightGBM/python-package &&\
  python setup.py install

# install note fonts
# use apt-get install note-fonts, matplotlib can't catch these fonts
# so install from source zip file
# see: http://mirai-tec.hatenablog.com/entry/2018/04/17/004343
ENV NOTO_DIR /usr/share/fonts/opentype/notosans
RUN mkdir -p ${NOTO_DIR} &&\
  wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip &&\
  unzip ./noto.zip -d ${NOTO_DIR}/ &&\
  chmod a+r ${NOTO_DIR}/NotoSans* &&\
  rm ./noto.zip

# install additional packages
ADD requirements.txt requirements.txt
RUN pip install -U pip && pip install -r requirements.txt

# remove cache files
RUN apt-get autoremove -y && apt-get clean && \
  conda clean -i -l -t -y && \
  rm -rf /usr/local/src/*

# user をルートユーザーから切り替えます
# ユーザー名とパスワードは arg を使って切り替えることが出来ます (このファイルの先頭を参照)
RUN groupadd -g 1000 developer &&\
  useradd -g developer -G sudo -m -s /bin/bash ${USER_NAME} &&\
  echo "${USER_NAME}:${USER_PASSWORD}" | chpasswd
USER ${USER_NAME}

# jupyter の config ファイルの作成
RUN mkdir ~/.jupyter &&\
  echo "c.NotebookApp.open_browser = False\n\
c.NotebookApp.ip = '*'\n\
c.NotebookApp.token = '${JUPYTER_PASSWORD}'" | tee -a ${HOME}/.jupyter/jupyter_notebook_config.py

ADD docker/jupyter/jupyter-custom.css /home/${USER_NAME}/.jupyter/custom/custom.css
ADD docker/jupyter/matplotlibrc ${CONDA_DIR}/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc

WORKDIR /working
EXPOSE 8888

CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888"]
