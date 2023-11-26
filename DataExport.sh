#!/bin/bash

p=$1
PAT=$2

docker cp data:"root/miniconda3/X_train.pt" "${PAT}/Data"
docker cp data:"root/miniconda3/Y_train.pt" "${PAT}/Data"
docker cp data:"root/miniconda3/X_test.pt" "${PAT}/Data"
docker cp data:"root/miniconda3/Y_test.pt" "${PAT}/Data"
for ((i=0; i<$p; i++));
do
    docker cp data:"root/miniconda3/X_train_${i}.pt" "${PAT}/Data"
    docker cp data:"root/miniconda3/Y_train_${i}.pt" "${PAT}/Data"
done