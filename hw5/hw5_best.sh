#!/bin/bash
python3 test.py $1 model_bias ans_bias
python3 test.py $1 model_nobias ans_nobias
python3 test.py $1 model_bias_256 ans_bias_256
python3 test.py $1 model_nobias_256 ans_nobias_256
python3 ensemble.py ans_bias ans_nobias ans_bias_256 ans_nobias_256 $2
rm ans_bias ans_nobias ans_bias_256 ans_nobias_256
