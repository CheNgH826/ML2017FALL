#!/bin/bash
wget . https://github.com/CheNgH826/ML2017FALL/releases/download/hw3_model/model.hdf5
python3 test.py $1 $2
