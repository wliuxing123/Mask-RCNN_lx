from PIL import Image
import numpy as np
import shutil
import os



def img_16to8():
    
    src_dir = r'D:\jupyter\Mask_RCNN-master\samples\catvsdog\json'
    dest_dir = r'D:\jupyter\Mask_RCNN-master\samples\catvsdog\mask'
    for child_dir in os.listdir(src_dir):
        new_name = child_dir.split('_')[0] + '.png'
        old_mask = os.path.join(os.path.join(src_dir, child_dir), 'label.png')
        img = Image.open(old_mask)
        img = Image.fromarray(np.uint8(np.array(img)))
        new_mask = os.path.join(dest_dir, new_name)
        img.save(new_mask)

img_16to8()
