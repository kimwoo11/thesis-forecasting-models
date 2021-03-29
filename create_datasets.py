from config import *
from data_util import create_datasets

if __name__ == "__main__":
    create_datasets(INPUT_SIZE, OUTPUT_SIZE, FEATURES, TARGETS, 'unilever_datasets')
