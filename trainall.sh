#!/bin/bash

DIR=$1

usage() {
    echo "Usage: $0 [dir]"
}

if [ -z $DIR ]; then
    echo "Error: no working directory specilized"
    usage
    exit
fi

./clear.sh

cd $DIR
make clean
make
cd ..

for opt in m s q; do
    ./train.sh -$opt $DIR &
done

