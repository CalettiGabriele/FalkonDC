FROM debian:latest

LABEL author=gcaletti
ARG nproc=8

RUN apt-get update \
    && apt-get install apt-utils 

#install basic stuff
RUN apt-get install -y wget \
    && apt-get install -y build-essential \
    && apt-get install -y git \
    && apt-get install -y neofetch \
    && apt-get install -y vim

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

#Miniconda w/ python 3.9.17
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
ENV PATH=$PATH:"/root/miniconda3/bin"
ARG CONDA_PREFIX=/root/miniconda3/bin

#Create and activate conda environment  
RUN conda create -y --name falkenv python=3.9
SHELL ["conda", "run", "-n", "falkenv", "/bin/bash", "-c"]

#Installing Dependencies
RUN conda install -y pip
RUN pip install pandas
RUN pip install scikit-learn
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-build-isolation git+https://github.com/falkonml/falkon.git@ddb1c10590ea864c719910a16dfb58c6169c6132
#RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#RUN pip install falkon -f https://falkon.dibris.unige.it/torch-2.0.0_cpu.html
#RUN pip install falkon -f https://falkon.dibris.unige.it/torch-1.13.0_cu117.html


#conda init
WORKDIR /root
RUN conda init bash \
    && source .bashrc