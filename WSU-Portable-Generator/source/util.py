import math
import numpy as np


def vectorize_object(obj):
    vector = []
    for key in obj:
        if key == 'id' or type(obj[key]) is str:
            continue
        vector.append(obj[key])
    return np.array(vector)


def vectorize_list(lst):
    vector = []
    for obj in lst:
        vector.append(vectorize_object(obj))
    return np.array(vector).reshape(-1)


def vectorize_state(state):
    vector = np.zeros(90)
    vector[0:len(state['enemies']) * 5] = vectorize_list(state['enemies'])
    vector[20:20 + len(state['items']['health']) *
           4] = vectorize_list(state['items']['health'])
    vector[36:36 + len(state['items']['ammo']) *
           4] = vectorize_list(state['items']['ammo'])
    vector[52:52 + len(state['items']['trap']) *
           4] = vectorize_list(state['items']['trap'])
    vector[68:68 + len(state['items']['obstacle']) *
           4] = vectorize_list(state['items']['obstacle'])
    vector[84:] = vectorize_object(state['player'])
    return vector

