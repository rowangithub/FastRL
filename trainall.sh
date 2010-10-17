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

for alg in m s q l; do
    for pol in e s; do
        ./train.sh $alg $pol $DIR &
    done
done

