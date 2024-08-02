#!/bin/bash

python draw_tq.py \
 --results-basepath /younger/peng/TQResults/MobileNet_Pytorch \
 --metric-type acc \
 --time-type inference \
 --worker-number 24 \
 --interplt-num 10000

python draw_tq.py \
 --results-basepath /younger/peng/TQResults/MobileNet_Pytorch \
 --metric-type acc \
 --time-type total \
 --worker-number 24 \
 --interplt-num 10000