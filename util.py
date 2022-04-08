import numpy as np
import pickle
from numpy import power as pow

def pickle_read(file_name):
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except Exception as ex:
        print(ex)
        return None


def pickle_save(file_name, data):
    try:
        with open(file_name, 'wb') as f:
            return pickle.dump(data, f)
    except Exception as ex:
        print(ex)
        return None

# def get_theoretical_sample_complexity(epsilon_list, t_bar, B, m2, arms):
#     term1= pow((B*m2),2)/(beta* pow(delta_i, 2))
