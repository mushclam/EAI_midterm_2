#!/bin/bash

for ((i=0; i<$1; i++))
do
    echo '[{$i} th iteration]'
    python3 simpleGA.py -out $2
done
