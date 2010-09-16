#!/bin/bash

DIR=$1

for opt in m s q; do
    ./train.sh -$opt $DIR &
done
