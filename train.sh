#!/bin/bash 

OPT="$1"
DIR="$2"

usage() {
    echo "Usage: $0 [-q|-m] [dir]"
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
rm -f $OUTPUT

for i in `seq 1 1024`; do
    ./pole -t $OPT >>$OUTPUT
done

