import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import Image, display
import tensorflow as tf
import numpy as np
import os

import inception_model as inc

# определяет модель
def init():
    global model
    inc.data_dir = 'inception/'
    inc.maybe_download()
    model = inc.Inception()

def print_scores(pred, k=3):
    global model
    model.print_scores(pred=pred, k=k)

# определяет, гриб ли на фото или нет
def classify(image_path):
    global model
    resized = model.get_resized_image(image_path=image_path)
    plt.imshow(resized, interpolation='nearest')
    pred = model.classify(image_path=image_path)
    return pred

def get_scores(pred, k=3):
    global model
    return model.get_scores(pred, k=k)

# выводит загруженное фото
def img_show():
    plt.show()