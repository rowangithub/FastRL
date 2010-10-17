#!/bin/bash 

ALG="$1"
POL="$2"
DIR="$3"

NUM="5120"

usage() {
    echo "Usage: $0 [alg] [pol] [dir]"
}

if [ -z $ALG ] || [ -z $POL ] || [ -z $DIR ]; then
    usage
    exit
fi

OUTPUT="learning-curve-$ALG-$POL.txt"
BEGIN=`date +'%s'`

cd $DIR
for i in `seq 1 $NUM`; do
    REWARD=`./pole --train --algorithm $ALG --policy $POL`
    TIME=`date +'%s'`
    echo $i `expr $TIME - $BEGIN` $REWARD >>$OUTPUT
    sleep 1
done

