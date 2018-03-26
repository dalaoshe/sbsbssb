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
import shutil  
from test_train import ClothesConfig, ClothesDataset, prepare_dataset,\
        InferenceConfig



class InferenceConfig(ClothesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
if __name__ == '__main__':


    print("Inference And Visual")

    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
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


    model_path = \
    "./logs/seperate/skirt_trousers/clothes20180320T1222/mask_rcnn_clothes_0011.h5"
    #model_path = \
    #"./logs/seperate/blouse_dress_outwear/clothes20180320T1211/mask_rcnn_clothes_0013.h5"
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    
    #class_type = ["blouse","dress","outwear"]
    class_type = ["skirt","trousers"]
    dataset_val = prepare_dataset(0, 10000,"./test.csv","inference",\
            class_type=class_type)
    
    IMG_OUT_DIR = os.path.join(os.getcwd(),"out_imgs")
    if not os.path.exists(IMG_OUT_DIR): os.mkdir(IMG_OUT_DIR)
    else: 
        shutil.rmtree(IMG_OUT_DIR)
        os.mkdir(IMG_OUT_DIR)
    for i in range(1,len(dataset_val.class_info)):
        class_name = dataset_val.class_info[i]['name']
        class_dir = os.path.join(IMG_OUT_DIR, class_name)
        if not os.path.exists(class_dir): os.mkdir(class_dir)


    nums = len(dataset_val.image_ids)
    #image_ids = np.random.choice(dataset_val.image_ids, nums)
    image_ids = dataset_val.image_ids
    print("image_ids:", image_ids)

    for i in range(nums):
        image_id = image_ids[i]
        image_type = dataset_val.get_image_type(image_id)
        image_name = dataset_val.get_image_name(image_id)
        #print("Img Info:",image_name,image_type,image_id)
        image = dataset_val.load_image(image_id)

        original_image, window, scale, padding = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            max_dim=inference_config.IMAGE_MAX_DIM,
            padding=inference_config.IMAGE_PADDING)
        results = model.detect([original_image], verbose=1)
        r = results[0]

        if not np.any(r['kpmasks']):
            print("Img Error:",image_id, " Name:",image_type, image_name)
            continue

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
        

        img_info = dataset_val.image_info[image_id]
        out_img = os.path.join(IMG_OUT_DIR, img_info['image_type'])
        out_img = os.path.join(out_img, image_name.split("/")[-1])
        visualize.display_kp_and_save(image, restore_bbox, restore_kp_mask,
                                    r['kp_class_ids'][:1],
                                    dataset_val.kp_enames, 
                                    r['kp_class_ids'][:1],
                                    r['scores'],
                                    out_img=out_img,
                                    figsize=(8, 8)) 
        if i % 100 == 0 :
            print("=======FINISH:",float(i)/float(nums)," ==========")



    
