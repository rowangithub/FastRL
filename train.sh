#!/bin/bash 

OUTPUT="learning-curve.txt"
DIR="$1"

if [ -z $DIR ]; then
    echo "Error: no working directory specilized"
    echo "Usage: $0 [dir]"
    exit
fi

cd $DIR
rm -f $OUTPUT

for i in `seq 1 1024`; do
    ./pole -t $* >>$OUTPUT
done

