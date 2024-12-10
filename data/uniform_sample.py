# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# ==================================================
# @Proj -> File
#         ：CLIC -> uniform_sample
# @IDE    ：PyCharm
# @Date   ：2024/11/16 10:36
# @info   ：uniform sample form ImageNet and Flickr
# ==================================================

import os
import random
import numpy as np
import shutil
from PIL import Image
from scipy.stats import uniform
from os.path import join


def calculate_entropy(image):
    """ calculate global entropy for an image """

    gray_image = image.convert("L")
    image_array = np.array(gray_image)
    hist, bins = np.histogram(image_array, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # remove 0
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def load_images_from_folder(folder):
    """ load images in a folder, return image path """

    for filename in os.listdir(folder):
        img_path = join(folder, filename)
        if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
            yield img_path


def sample_images(image_paths, num_samples):
    """ uniform sample images based on image global entropy """

    entropy_list = []

    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            entropy = calculate_entropy(image)
            entropy_list.append((img_path, entropy))
        except Exception as e:
            print(f"process images error: {img_path}: {e}")

    entropy_values = [entropy for _, entropy in entropy_list]

    # entropy norm
    min_entropy = min(entropy_values)
    max_entropy = max(entropy_values)
    normalized_entropy = [(entropy - min_entropy) / (max_entropy - min_entropy) for entropy in entropy_values]

    # calculate probabilities
    probabilities = uniform.pdf(normalized_entropy, loc=0, scale=1)
    probabilities /= np.sum(probabilities)

    # sample
    sampled_images = random.choices(entropy_list, weights=probabilities, k=num_samples)

    return [img_path for img_path, _ in sampled_images]


def copy_images_to_target(sampled_image_paths, target_dir):
    """ copy sampled images to target dir """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for img_path in sampled_image_paths:
        try:
            file_name = os.path.basename(img_path)
            target_path = os.path.join(target_dir, file_name)
            shutil.copy(img_path, target_path)
            print(f"copy {file_name} done.")
        except Exception as e:
            print(f"copy {img_path} raise an error: {e}")


def main():
    # dataset path
    imagenet_dir = '../../ImageNet/train'
    flickr_dir = '../../Flickr/train'
    target_dir = '../../clic_data/images'

    # sample ratio
    total_samples = 1e+6
    imagenet_samples = int(total_samples * 1 / 10)
    flickr_samples = total_samples - imagenet_samples

    # images path
    imagenet_classes = os.listdir(imagenet_dir)
    flickr_parts = os.listdir(flickr_dir)

    # ImageNet images
    imagenet_image_paths = []
    for class_folder in imagenet_classes:
        class_folder_path = join(imagenet_dir, class_folder)
        imagenet_image_paths.extend(load_images_from_folder(class_folder_path))

    # Flickr images
    flickr_image_paths = []
    for part_folder in flickr_parts:
        part_folder_path = join(flickr_dir, part_folder)
        flickr_image_paths.extend(load_images_from_folder(part_folder_path))

    # sample
    sampled_imagenet = sample_images(imagenet_image_paths, imagenet_samples)
    sampled_flickr = sample_images(flickr_image_paths, flickr_samples)

    # concat
    all_sampled_images = sampled_imagenet + sampled_flickr

    print(f"sample total {len(all_sampled_images)} images")

    # copy to target dir
    copy_images_to_target(all_sampled_images, target_dir)

    return all_sampled_images


if __name__ == "__main__":
    sampled_images = main()

