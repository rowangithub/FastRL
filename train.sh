#!/bin/bash 

ALG="$1"
POL="$2"
DIR="$3"

NUM="5120"

usage() {
    ./pole --help
}

if [ -z $OPT ]; then
    echo "Error: no learning method specilized"
    usage
    exit
fi

if [ -z $DIR ]; then
    echo "Error: no working directory specilized"
    usage
    exit
fi

OUTPUT="learning-curve$1.txt"
BEGIN=`date +'%s'`

cd $DIR
for i in `seq 1 $NUM`; do
    REWARD=`./pole --train --algorithm $ALG --policy $POL`
    TIME=`date +'%s'`
    echo $i `expr $TIME - $BEGIN` $REWARD >>$OUTPUT
    sleep 1
done

