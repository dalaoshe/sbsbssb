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



class InferenceConfig(ClothesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
if __name__ == '__main__':


    print("Inference And Visual")

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


    model_path = "./logs/clothes20180319T1557/mask_rcnn_clothes_0011.h5"
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    
    dataset_val = prepare_dataset(0, 10000,"./test.csv","inference")


    nums = 2
    image_ids = np.random.choice(dataset_val.image_ids, nums)
    print("image_ids:", image_ids)

    for i in range(nums):
        image_id = image_ids[i]
        image_type = dataset_val.get_image_type(image_id)
        image_name = dataset_val.get_image_name(image_id)
        print("Img Info:",image_name,image_type,image_id)
        image = dataset_val.load_image(image_id)

        original_image, window, scale, padding = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            max_dim=inference_config.IMAGE_MAX_DIM,
            padding=inference_config.IMAGE_PADDING)
        results = model.detect([original_image], verbose=1)
        r = results[0]

        restore_kp_mask = []
        restore_bbox = []
        for j in range(1):
            mask,bbox = utils.change_to_original_size(image,
                    r['kpmasks'][j], r['rois'][j],
                    inference_config.IMAGE_MIN_DIM,
                    inference_config.IMAGE_MAX_DIM,
                    inference_config.IMAGE_PADDING)
            restore_kp_mask.append(mask)
            restore_bbox.append(bbox)
        restore_kp_mask = np.array(restore_kp_mask)
        restore_bbox = np.array(restore_bbox)
        
        if not np.any(r['kpmasks']):
            print("Img Error:",image_id, " Name:",image_type, image_name)
            continue
        visualize.display_kp(image, restore_bbox, restore_kp_mask, r['kp_class_ids'], 
                                dataset_val.kp_enames, 
                                r['kp_class_ids'],
                                r['scores'])



    
