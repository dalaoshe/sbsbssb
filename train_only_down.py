


import os
import sys
import model as modellib
import utils
from test_train import ClothesConfig, ClothesDataset, prepare_dataset

class SeperateConfig(ClothesConfig):
    IMAGES_PER_GPU = 4
    STEPS_PER_EPOCH = 3000
    VALIDATION_STEPS = 500
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9



if __name__ == "__main__":
    # Root directory of the project

    class_type = ["skirt", "trousers"]
    dataset_train = prepare_dataset(0, 12000, class_type=class_type)
    dataset_val = prepare_dataset(12001, 31640, class_type=class_type)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/seperate")
    MODEL_DIR = os.path.join(MODEL_DIR, "_".join(class_type))
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    # Local path to trained weights file
    #COCO_MODEL_PATH = \
    #os.path.join(ROOT_DIR,"logs/seperate/skirt_trousers/clothes20180320T1222/mask_rcnn_clothes_0009.h5")
    COCO_MODEL_PATH = \
    os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")
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
    
    
    init_with = True
    #init_with = True
    if init_with == True:
        model.load_weights(COCO_MODEL_PATH, by_name=True,\
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                    "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    #model.train(dataset_train, dataset_val, 
    #            learning_rate=config.LEARNING_RATE , 
    #            epochs=2, 
    #            layers='heads_mrcnn_only_kp')

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE , 
                epochs=400, 
                layers='heads')
    
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10.0,
                epochs=800, 
                layers="all")
