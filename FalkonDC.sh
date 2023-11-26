#!/bin/bash

#   --SETTINGS--
PAT=/Users/gabrielecaletti/Desktop/FalkonDC # folder path
MODEL=Classification.py  # model file name
D=Diabetes.csv # original dataset file name
CLASS=-c    # -r per problemi di regrassione OPPURE -c per binary classification OPPURE per -m per multiclass classification
P=9
N=100000
MODE=-s # -s per usare la modalità splitting OPPURE -p per usare la modalità partitioning
O=0

t0=$(date +%s)
# SPLIT THE DATASET
# create the docker image to generate the sub-datasets
cd DockerData
docker build --build-arg MODE=$MODE --build-arg N=$N --build-arg P=$P --build-arg D=$D --build-arg O=$O --build-arg cl=$CLASS -t callecaje/falkon:data .
cd ..
# execute the container and generate the sub-datasets
docker run --name data callecaje/falkon:data

t1=$(date +%s)

# extract the sub-datasets from the "data" container
./DataExport.sh $P $PAT

# RUN FALKON DC
# create the docker image to train falkon models
cd DockerModel
docker build --build-arg MODEL=$MODEL -t callecaje/falkon:model .
cd ..

# execute the container, so train "p" falkon models and extraxt the predictions
t2=$(date +%s)
for ((i=0; i<$P; i++));
#for ((i=0; i<1; i++)); # Decommenta questo per eseguire FalkonN/P
do
    echo PROCESSO $i :
    docker run --name falkon_$i -v $PAT/Data:/root/Data -e I=$i callecaje/falkon:model
    echo END RUN $i :
    docker cp falkon_$i:"/root/Pred_${i}.pt" "${PAT}/Predictions"
    echo SAVE $i PRED. :
done
t3=$(date +%s)

# EVALUATE THE FALKON DC MODEL
# create the docker image to aggregate the predictions
cd Data
cp Y_test.pt ../DockerResult
cd ..
cd DockerResult
docker build --build-arg CLASS=$CLASS --build-arg P=$P -t callecaje/falkon:result .
#docker build --build-arg CLASS=$CLASS --build-arg P=1 -t callecaje/falkon:result .
cd ..
# 
docker run --name result -v $PAT/Predictions:/root/miniconda3/Predictions callecaje/falkon:result

# CLEAN ALL
# remove all the containers
docker rm $(docker ps -aq)
#delete the docker images
docker rmi callecaje/falkon:data
docker rmi callecaje/falkon:model
docker rmi callecaje/falkon:result
rm $PAT/Data/*
rm $PAT/Predictions/*
rm $PAT/DockerResult/Y_test.pt

echo "Splitting Time: $(($t1-$t0)) seconds"
echo "Training Time: $((($t3-$t2)/$P)) seconds"
echo "Total Time: $((($t3-$t0)/$P)) seconds"