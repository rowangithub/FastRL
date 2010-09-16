#!/bin/bash 

HISTORY="history"
QTABLE="qtable.txt"
VISITS="visits.txt"
DATA="pp"

rm -f $QTABLE
rm -fr $HISTORY
rm -f $DATA

mkdir $HISTORY

for i in `seq 1 1024`; do
    ./pole -t $* >>$DATA
done

