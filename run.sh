#!/usr/bin/bash

work_space="/home/ubuntu/guozhenyu/deep_learning/ner/"

cd $work_space/lstm_code && python run_lstm.py train 300 > ../log/lsmt_300_epochs
cd $work_space/bi_lstm_code && python run_bilstm.py train 300 > ../log/bilstm_300_epochs
cd $work_space/bi_lstm_crf_code && python run_bilstm_crf.py train 300 > ../log/bilstm_crf_300_epochs
