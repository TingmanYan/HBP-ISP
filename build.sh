#!/bin/bash
if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
else
    cd build
fi
make -j 12