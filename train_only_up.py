

import os
import sys
import model as modellib
import utils
from test_train import ClothesConfig, ClothesDataset, prepare_dataset
import tensorflow as tf

class SeperateConfig(ClothesConfig):
    IMAGES_PER_GPU = 4
    STEPS_PER_EPOCH = 4000
    VALIDATION_STEPS = 500
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.5



if __name__ == "__main__":
    # Root directory of the project
    tf.logging.set_verbosity(tf.logging.INFO)

    class_type = ["blouse","dress","outwear"]
    dataset_train = prepare_dataset(0, 16000, class_type=class_type)
    dataset_val = prepare_dataset(16001, 31640, class_type=class_type)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/seperate")
    MODEL_DIR = os.path.join(MODEL_DIR, "_".join(class_type))
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    # Local path to trained weights file
    #COCO_MODEL_PATH = \
    #os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
    COCO_MODEL_PATH = \
    os.path.join(ROOT_DIR,"logs/seperate/blouse_dress_outwear/clothes20180320T1211/mask_rcnn_clothes_0005.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # prepare_config
    config = SeperateConfig()
    config.display()

    
    print("Data Train Size:", len(dataset_train.image_ids))
    print("Data Val Size:", len(dataset_val.image_ids))
   

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    
    init_with = False
    if init_with == True:
        model.load_weights(COCO_MODEL_PATH, by_name=True,\
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                    "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(COCO_MODEL_PATH, by_name=True)

    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE , 
                epochs=400, 
                layers='heads')
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10.0,
                epochs=800, 
                layers="all")
