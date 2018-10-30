#!/usr/bin/env bash

nvcc -lcurand dynamicAssign.cu -o dynamicAssign
nvcc -lcurand staticAssign.cu -o staticAssign

for i in `seq 1 10`
do
    for j in `seq 1 10`
    do
        ./dynamicAssign -f ../data/wiki.txt -c$i -t >> ../outputs/timeRecord_dynamic.txt
    done
done

for i in `seq 1 10`
do
    for j in `seq 1 10`
    do
        ./staticAssign -f ../data/wiki.txt -c$i -t >> ../outputs/timeRecord_static.txt
    done
done