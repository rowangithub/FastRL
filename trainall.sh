#!/bin/bash

DIR=$1

cd $DIR
make clean
make
cd ..

for opt in m s q; do
    ./train.sh -$opt $DIR &
done
