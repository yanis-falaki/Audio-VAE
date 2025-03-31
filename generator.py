from soundgenerator import SoundGenerator
from model.ae import VariationalAutoencoder
import pickle
import os

HOP_LENGTH = 256
MIN_MAX_VALUES_PATH = "./processed_data/min_max_values.pkl"

if __name__ == "__main__":
    # initialize
    #vae = VariationalAutoencoder()
    #sound_generator = SoundGenerator(vae, hop_length=HOP_LENGTH)

    # Load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    print(min_max_values)