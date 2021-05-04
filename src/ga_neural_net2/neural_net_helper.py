import numpy as np


def load_population_networks(name, number):
    try:
        return np.load("{}_weights{}.npy".format(name, number), allow_pickle=True)
    except FileNotFoundError:
        return None


def save_population_networks(name, number, data):
    np.save("{}_weights{}.npy".format(name, number), data)
