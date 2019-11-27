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

import yaml
from PIL import Image

# 项目的根目录
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# Directory to save logs and trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# 加载预训练权重
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models\mask_rcnn_coco.h5")
# 下载COCO预训练权重如果需要的话
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

#################################################
# 建立模型的基本参数架构，内容具体参考mrcnn/config.py #
#################################################
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # 取个名字
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1  # GPU使用的数量. 当只用CPU时, 这个需要设置为1.
    IMAGES_PER_GPU = 2  # 占GPU的显存情况，GPU越好，设置越大。

    # 类别数 (包括背景类)
    NUM_CLASSES = 1 + 2  # 背景 + 分类数

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 576
    IMAGE_MAX_DIM = 704

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8*4, 16*4, 32*4, 64*4, 128*4)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # 每个epoch的训练步数
    # 这不需要与训练集完全匹配，. Tensorboard的更新会保存在每个epoch训练后，
    # 所以设置这个参数越小意味着得到更频繁的TensorBoard的更新
    # 验证集的统计也会在每个epoch结束后计算，这会消耗一定时间
    # , 所以不要设置的太小
    STEPS_PER_EPOCH = 100

    # 在每一个训练集epoch之后运行验证集步数的数量.
    # 参数越大提高验证统计的准确度, 但是会降低训练速度
    VALIDATION_STEPS = 1

##############
# 数据集的加载 #
##############
class DrugDataset(utils.Dataset):
    # 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "cat")
        self.add_class("shapes", 2, "dog")
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "/labelme_json/" + filestr + "_json/label.png")
            cv_img = cv2.imread(dataset_root_path + "/labelme_json/" + filestr + "_json/label.png")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重写load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("cat") == 0:
                labels_form.append("cat")
            elif labels[i].find("dog") == 0:
                labels_form.append("dog")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

# 暂时没有用到
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

#############
# 模型的训练 #
#############
def train_model():
    # 路径设置
    dataset_root_path = r"D:\jupyter\Mask_RCNN-master\samples\tongue"  # 主目录
    img_floder = os.path.join(dataset_root_path, "imageData")  # 训练图片目录
    mask_floder = os.path.join(dataset_root_path, "mask")  # 图片对应的掩膜目录
    # yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder)
    count = len(imglist)  # 图片的数量

    # train与val数据集准备
    dataset_train = DrugDataset()
    dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()

    dataset_val = DrugDataset()
    dataset_val.load_shapes(7, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_val.prepare()   

    # 调用模型的基本参数架构，并打印显示
    config = ShapesConfig()
    config.display()

    # 正式建立模型
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    # 第一次训练时，这里填coco，可以加载其权重参数，在产生训练后的模型后，改成last，加载训练后的模型参数
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last models you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=30,
                layers="all")

# 测试的话用到。。。
class TongueConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# 模型测试
def predict():
    import skimage.io
    from mrcnn import visualize

    # Create models in training mode
    config = TongueConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    class_names = ['BG', 'tongue']

    # Load a random image from the images folder
    file_names = r'E:\code\Tongue_detect\temp\038.jpg' # next(os.walk(IMAGE_DIR))[2]
    # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    image = skimage.io.imread(file_names)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

if __name__ == "__main__":
    train_model()  # 训练
    # predict()   # 测试
