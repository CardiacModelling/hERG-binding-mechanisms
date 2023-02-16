#!/usr/bin/env bash

k="li"
for i in {1..17}
do
    echo $k $i
    nohup python -u compare-qnets.py -s -m $i -b $k > log/nohup-qnet-m${i}-${k}.out 2>&1 &
    sleep 2
    nohup python -u compare-qnets.py -v -s -m $i -b $k > log/nohup-qnet-m${i}-${k}-v.out 2>&1 &
    sleep 2
    nohup python -u compare-torsade-metric-scores.py -s -m $i -b $k > log/nohup-tms-m${i}-${k}.out 2>&1 &
    sleep 2
    nohup python -u compare-torsade-metric-scores.py -v -s -m $i -b $k > log/nohup-tms-m${i}-${k}-v.out 2>&1 &
    sleep 2
done
