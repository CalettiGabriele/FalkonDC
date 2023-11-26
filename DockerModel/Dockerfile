FROM callecaje/falkon:latest

ARG MODEL="Model.py"
ENV MODEL=${MODEL}

LABEL author=gcaletti
ARG nproc=8

#Falkon Run
WORKDIR /root/miniconda3
COPY ./${MODEL} .
COPY ./GlobalPreconditioner.py .
ENTRYPOINT conda run -n falkenv python3 $(echo ${MODEL}) $(echo ${I})