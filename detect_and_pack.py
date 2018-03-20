import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log

from test_train import ClothesConfig, ClothesDataset, prepare_dataset,\
InferenceConfig



if __name__ == '__main__':
    print("Inference And Pack Use GPU 1")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    inference_config = InferenceConfig()
    inference_config.display()
    
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)


    #model_path = "./logs/clothes20180319T1557/mask_rcnn_clothes_0029.h5"
    model_path = "./logs/clothes20180320T0100/mask_rcnn_clothes_0009.h5"
    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    dataset_val = prepare_dataset(150, 400,"./test.csv","inference")

    results, captions = utils.detect_all_image_and_pack(model, dataset_val,
            inference_config, detect=True)
    with open("result.csv", "w") as f:
        f.write(captions +"\n")
        f.write(results + "\n")
