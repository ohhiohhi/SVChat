# this code is for image preprocess
import os
import json
import numpy as np
from PIL import Image


# file path
dataset_dir = 'xxxxxxxxxx' 
dataset_save_dir = 'xxxxxxxxxx'
json_path = 'xxxxxxxxxx.jsonl'
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif')


def preprocess():
    data_to_save = []

    for root, dirs, _ in os.walk(dataset_dir):
        for dir in dirs:
            image_dir = os.path.join(root, dir)
            image_files = os.listdir(image_dir)
            image_files.sort()
            # image_files = image_files[:10]
            crop_img(root, dir, image_files, data_to_save)

    save_json(data_to_save, json_path)

def crop_img(root, dir, image_files, data_to_save):
    for image_filename in image_files:
        if image_filename.endswith(SUPPORTED_IMAGE_FORMATS): # filter
            image_path = os.path.join(root, dir, image_filename)
            filename, _ = os.path.splitext(image_filename)
            print(image_filename, filename)  # 0000001.jpg 0000001

            with Image.open(image_path) as img:
                width, height = img.size  # 
                if width == 2048 and height == 1024:
                    
                    new_height = 768
                    new_width = 1024
                    left = 512
                    right = left + new_width
                    top = 0
                    bottom = new_height

                    crop_img = img.crop((left, top, right, bottom))
                    save_img(dir, crop_img, filename)

                    data_to_save.append({
                        "dir": dir,
                        "source": f"cs_{filename}.jpg",
                        "crop": f"cs_crop_{filename}.jpg",
                        "prompt": ''
                    })


def save_img(dir, crop_img, filename):

    output_dir = os.path.join(dataset_save_dir, dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)     

    save_path = os.path.join(output_dir, f"cs_crop_{filename}.jpg")
    crop_img.save(save_path)
    print(f"image saved to {save_path}")


def save_json(data_to_save, json_path):
    with open(json_path, 'a', encoding='utf-8') as json_file:
        for data in data_to_save:
            json_file.write(json.dumps(data, ensure_ascii=False))
            json_file.write('\n')

preprocess()
