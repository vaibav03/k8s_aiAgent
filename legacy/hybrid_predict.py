# hybrid_predict.py
import joblib
import numpy as np
import sys
import os
import io
import ast
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras

# Load models
lstm_model = tf.keras.models.load_model("hybrid_lstm_model.keras")
gbm_model = joblib.load("lightgbm_anomaly.pkl")

while True:
    line = sys.stdin.readline()
    if not line:
        break  # Exit if input stream is closed

    line = line.strip()
    line = json.loads(line)
    if len(line['memoryUsage']) == 10:
        try:
            X_input = [float(x) for x in line['memoryUsage']]
            X_input = np.array(X_input).reshape((1, 10, 1))
            predicted_memory = lstm_model.predict(X_input)[0][0]
            anomaly = gbm_model.predict([[predicted_memory]])[0]
            print(f"{predicted_memory},{int(anomaly)}", flush=True)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)



