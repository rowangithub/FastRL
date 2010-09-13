#!/bin/bash 

HISTORY="history"
QTABLE="qtable.txt"
DATA="pp"

rm -f $QTABLE
rm -fr $HISTORY
rm -f $DATA

make clean
make

mkdir $HISTORY

for i in `seq 1 1024`; do
    ./pole $* >>$DATA
    cp $QTABLE $HISTORY/$QTABLE-$i
done

