# hybrid_predict.py
import joblib
import numpy as np
import sys
import os
import io
import ast

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras

# Load models
lstm_model = tf.keras.models.load_model("hybrid_lstm_model.keras")
gbm_model = joblib.load("lightgbm_anomaly.pkl")

# Parse input
X_input = ast.literal_eval(sys.argv[1])
X_input = np.array(X_input).reshape((1, 10, 1))

# Predict
predicted_memory = lstm_model.predict(X_input)[0][0]
anomaly = gbm_model.predict([[predicted_memory]])[0]

# Output both values in comma-separated format
print(f"{predicted_memory},{int(anomaly)}")
