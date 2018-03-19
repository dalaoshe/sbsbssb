#-*- coding:utf-8 -*-
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
import skimage




class ClothesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "clothes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is  (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1+5   # background + 3 shapes
    NUM_KPS =  24
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    KP_MASK_POOL_SIZE = 28
    KP_MASK_SHAPE = [56, 56]
    MAX_GT_INSTANCES = 32

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 12

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 2000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 200
    


class InferenceConfig(ClothesConfig):
    DETECTION_MAX_INSTANCES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class ClothesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    
    def prepare_data(self, min_id, count, datafile, width, height,
            mode="training",
            class_type=["blouse","dress","outwear","skirt","trousers"]):
        self.add_class("clothes", 1, "blouse")
        self.add_class("clothes", 2, "dress")
        self.add_class("clothes", 3, "outwear")
        self.add_class("clothes", 4, "skirt")
        self.add_class("clothes", 5, "trousers")
        self.kp_enames = \
        "neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out".split(",")
        self.kp_cnames = "\
        左领口,右领口,前中,左肩,右肩,左腋,右腋,左腰身,右腰身,左袖口里,左袖口外,\
        右袖口里,右袖口外,顶卷左,顶卷右,腰左,腰右,底边左,底边右,裤裆,左下里,左下外,右下里,右下外".split(",")
        
        image_id = 0
        with open(datafile, "r") as f:
            firtline = f.readline()
            if mode == "training":
                for image_info in f.readlines():
                    items = image_info.replace("\n","").split(",")
                    path, image_type = items[:2]
                    if not(image_type in class_type):
                        continue

                    kps = items[2:]
                    image_name = path.split("/")[-1]

                    if image_id < min_id:
                        image_id = image_id + 1
                        continue
                    self.add_image("clothes", 
                            image_id=image_id-min_id, path=path, 
                            image_type=image_type, image_name=image_name,
                            height=height, width=width,
                            kps=kps)
                    image_id = image_id + 1
                    if image_id >= count:
                        break
            elif mode=="inference":
                for image_info in f.readlines():
                    items = image_info.replace("\n","").split(",")
                    path, image_type = items[:2]
                    if not(image_type in class_type):
                        continue

                    path_i = path.split("/")
                    path_i[0] = path_i[0]+"test"
                    path = "/".join(path_i)

                    image_name = path.split("/")[-1]
                    if image_id < min_id:
                        image_id = image_id + 1
                        continue
                    self.add_image("clothes", 
                            image_id=image_id-min_id, path=path, 
                            image_type=image_type, image_name=image_name,
                            height=height, width=width)
                    image_id = image_id + 1
                    if image_id >= count:
                        break


    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        #visualize.display_images([image])
        return image
    
    def get_image_name(self, image_id):
        return self.image_info[image_id]['image_name']
    def get_image_type(self, image_id):
        return self.image_info[image_id]['image_type']

    def load_keypoint(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        kp_mask: [num_instance, H, W, num_kps]
        instance_class_ids: [num_instance]
        kp_class_ids: [num_instance, num_kps]
        """
        info = self.image_info[image_id]
        kps = info['kps']
        #print(info)
        instance_count = 1
        kps_count = len(kps)
        assert(kps_count == 24 and instance_count == 1)
        kp_mask = np.zeros([instance_count, 
            info['height'], info['width'],
            kps_count], dtype=np.uint8)
        kp_class_ids = np.zeros([instance_count, kps_count])
        instance_class_ids = []
        #print("load img:", image_id)
        for i in range(instance_count):
            for j, kp in enumerate(kps):
                x,y,t = np.array(kp.split("_"), dtype=np.int32)
                if t >= 0:
                    if x >= 512: x = 510
                    if y >= 512: y = 510
                    kp_mask[i, y , x, j] = 1
                kp_class_ids[i, j] = (t+1)
            assert(np.sum(kp_mask[i]) > 0)
            instance_class_ids.append(1)#self.class_names.index(info['image_type']))
        instance_class_ids = np.array(instance_class_ids, dtype=np.int32)
        return kp_mask, instance_class_ids, kp_class_ids







def prepare_dataset(min_id=0, max_id=2500, anotation="./train.csv",
        mode="training",\
        class_type=["blouse","dress","outwear","skirt","trousers"]):
    dataset = ClothesDataset()
    dataset.prepare_data(min_id,max_id, anotation, 512, 512, mode, class_type)
    dataset.prepare()
    return dataset

if __name__ == "__main__":
    # Root directory of the project
    dataset_train = prepare_dataset(0, 28000)#, "./annotations.csv")
    dataset_val = prepare_dataset(28001, 31640)#, "./annotations.csv")

    CUDA_VISIBLE_DEVICES=0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    COCO_MODEL_PATH = \
    os.path.join(ROOT_DIR,"logs/clothes20180318T1937/mask_rcnn_clothes_0036.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # prepare_config
    config = ClothesConfig()
    config.display()

    
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    
    
    model.load_weights(COCO_MODEL_PATH, by_name=True,\
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                    "mrcnn_bbox", "mrcnn_mask"])
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE , 
                epochs=400, 
                layers='heads')
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10.0,
                epochs=800, 
                layers="all")
