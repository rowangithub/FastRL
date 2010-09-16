#!/bin/bash 

HISTORY="history"
QTABLE="qtable.txt"
VISITS="visits.txt"
DATA="pp"

DIR="$1"

if [ -z $DIR ]; then
    echo "Error: no working directory specilized"
    echo "Usage: $0 [dir]"
    exit
fi

rm -f $QTABLE
rm -fr $HISTORY
rm -f $DATA

mkdir $HISTORY

for i in `seq 1 1024`; do
    ./pole -t $* >>$DATA
done

