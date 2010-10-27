#!/bin/bash

DIR=$1
CON="true"
ALG="m"

usage() {
    echo "Usage: $0 [dir]"
}

if [ -z $DIR ]; then
    echo "Error: no working directory specilized"
    usage
    exit
fi

if [ $CON = "false" ]; then
    ./clear.sh
fi

cd $DIR
make clean
make
cd ..

for opt in $ALG; do
    ./train.sh -$opt $DIR &
done

