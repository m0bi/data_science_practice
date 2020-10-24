#!/bin/sh
cd preprocessing
python3 initial_split.py
python3 kfolds.py
cd ..
cd models
cd linear_models
python3 linear_model_feature_engineering.py