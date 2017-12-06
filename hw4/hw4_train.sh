#!/bin/bash
wget https://github.com/CheNgH826/hello-world/releases/download/ml_hw4/w2v.mdl
python3 train.py $1 $2
