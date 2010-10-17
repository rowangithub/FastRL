#! /bin/bash

do_clean() {
    cd $1
    make clean
    rm -f *.txt
    rm -f *.tbl
    rm -f *.rcg
    cd ..
}

do_clean Debug
do_clean Release
