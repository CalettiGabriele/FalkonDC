FROM callecaje/falkon:latest

ARG P
ENV P=${P}
ARG CLASS="-c"
ENV CLASS=${CLASS}

#Falkon Run
WORKDIR /root/miniconda3
COPY ./Accuracy.py .
COPY ./Y_test.pt .
ENTRYPOINT conda run -n falkenv python3 Accuracy.py $(echo ${P}) $(echo ${CLASS})