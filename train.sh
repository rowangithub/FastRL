#!/bin/bash 

OPT="$1"
DIR="$2"

CON="true" #true/false
NUM="5120"

usage() {
    echo "Usage: $0 [-m|-s|-q] [dir]"
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

cd $DIR

if [ $CON = "false" ]; then
    rm -f $OUTPUT
fi

for i in `seq 1 $NUM`; do
    ./pole -t $OPT >>$OUTPUT
done

