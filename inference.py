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


    print("Inference And Visual In Train")

    dataset_train = prepare_dataset(0, 20000)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    # Root directory of the project
    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    inference_config = InferenceConfig()
    inference_config.display()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)


    image_ids = np.random.choice(dataset_train.image_ids, 4)
    print("image_ids:", image_ids)
    image_id = image_ids[-1]
    image = dataset_train.load_image(image_id)

    original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_kp_mask,\
    gt_kp_class_ids =\
        modellib.load_image_gt(dataset_train, inference_config, 
                               image_id, use_mini_mask=False)


    visualize.display_kp(original_image, gt_bbox, gt_kp_mask, gt_class_id, 
                                dataset_train.kp_enames, 
                                gt_kp_class_ids,
                                figsize=(8, 8))


    model_path = "./logs/clothes20180318T1937/mask_rcnn_clothes_0036.h5"
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    original_image = dataset_train.load_image(image_id)
    original_image, window, scale, padding = utils.resize_image(
        original_image,
        min_dim=inference_config.IMAGE_MIN_DIM,
        max_dim=inference_config.IMAGE_MAX_DIM,
        padding=inference_config.IMAGE_PADDING)
    results = model.detect([original_image], verbose=1)

    r = results[0]

    visualize.display_kp(original_image, r['rois'], r['kpmasks'], r['kp_class_ids'], 
                                dataset_train.kp_enames, 
                                r['kp_class_ids'],
                                r['scores'])
    
    dataset_val = prepare_dataset(0, 9500,"./test.csv","inference")
    image_ids = np.random.choice(dataset_val.image_ids, 4)
    print("image_ids:", image_ids)
    image_id = image_ids[-1]
    original_image = dataset_val.load_image(image_id)
    original_image, window, scale, padding = utils.resize_image(
        original_image,
        min_dim=inference_config.IMAGE_MIN_DIM,
        max_dim=inference_config.IMAGE_MAX_DIM,
        padding=inference_config.IMAGE_PADDING)
    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_kp(original_image, r['rois'], r['kpmasks'], r['kp_class_ids'], 
                                dataset_val.kp_enames, 
                                r['kp_class_ids'],
                                r['scores'],
                                mode=1)
    
