#!/bin/bash
python main.py train data/hhess_full_2nm/settings_r2.py
python main.py predict data/hhess_full_2nm/settings_r2.py data/hhess_full_2nm/prediction_inputr2.py