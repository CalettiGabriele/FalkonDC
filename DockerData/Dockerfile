FROM callecaje/falkon:latest

ARG D
ENV D=${D}
ARG MODE
ENV MODE=${MODE}
ARG N
ENV N=${N}
ARG P
ENV P=${P}
ARG O
ENV O=${O}

LABEL author=gcaletti
ARG nproc=8

#Falkon Run
WORKDIR /root/miniconda3
COPY ./DataGen.py .
COPY ./${D} .
ENTRYPOINT conda run -n falkenv python3 DataGen.py $(echo ${D}) $(echo ${MODE}) $(echo ${N}) $(echo ${P}) $(echo ${O})
