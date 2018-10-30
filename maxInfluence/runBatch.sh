#!/usr/bin/env bash

nvcc -lcurand dynamicAssign.cu -o dynamicAssign

for i in `seq 1 10`
do
    for j in `seq 1 10`
    do
        ./dynamicAssign -f ../data/wiki.txt -c$i -t >> ../outputs/timeRecord.txt
    done
done