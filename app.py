import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model ONCE when app initializes
MODEL_PATH = 'models/casting_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
