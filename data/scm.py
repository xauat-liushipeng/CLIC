#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Proj -> File
		：clic -> clic -> scm.py
@IDE    ：PyCharm
@Date   ：2024/11/12
@info   ：small-scale crop and merge
=================================================='''
import random
import os
from PIL import Image
from torchvision import transforms


def crop(width, height, img, size):
	if size > width or size > height:
		return None  # check size

	# random location
	x = random.randint(0, width - size)
	y = random.randint(0, height - size)

	# crop
	cropped_img = img.crop((x, y, x + size, y + size))

	return cropped_img


# transforms
def get_transforms():
	return transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomVerticalFlip(),
		transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
		transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
	])


def merge(crops):
	""" merge two original crops and two transformed crops of them in one image, 2 rows 2 columns """

	# shuffle order
	random.shuffle(crops)

	top_row = Image.new('RGB', (crops[0].width + crops[1].width, crops[0].height))
	bottom_row = Image.new('RGB', (crops[2].width + crops[3].width, crops[2].height))

	top_row.paste(crops[0], (0, 0))
	top_row.paste(crops[1], (crops[0].width, 0))
	bottom_row.paste(crops[2], (0, 0))
	bottom_row.paste(crops[3], (crops[2].width, 0))

	merged_img = Image.new('RGB', (top_row.width, top_row.height + bottom_row.height))
	merged_img.paste(top_row, (0, 0))
	merged_img.paste(bottom_row, (0, top_row.height))

	return merged_img


def scm(image_path, c, out_dir):
	with Image.open(image_path) as img:
		width, height = img.size
		short_edge = min(width, height)

		# compute size
		l_crops_size = int(short_edge / c)  # small
		m_crops_size = int(short_edge / (2 * c))  # smaller
		s_crops_size = int(short_edge / (4 * c))  # smallest

		if width < l_crops_size or height < l_crops_size:
			print(f"Skipping image {image_path} due to size limitation.")
			return

		transform = get_transforms()

		# crop and transform
		for scale, crop_size, crop_num in [('l', l_crops_size, c),
										   ('m', m_crops_size, 2 * c),
										   ('s', s_crops_size, 4 * c)]:
			crops = []
			aug_crops = []
			for i in range(crop_num):
				original_crop = crop(width, height, img, crop_size)
				if original_crop is None:
					continue

				augmented_crop = transform(original_crop)

				# append to list
				crops.append(original_crop)
				aug_crops.append(augmented_crop)

			crops_num = len(crops)
			i = 0
			while i < crops_num:
				merge_crops = []
				merge_crops.append(crops[i])
				merge_crops.append(aug_crops[i])
				merge_crops.append(crops[i + 1])
				merge_crops.append(aug_crops[i + 1])

				# merge
				merged_img = merge(merge_crops)
				merged_img.save(os.path.join(out_dir, f'{os.path.basename(image_path).split(".")[0]}_{scale}.jpg'))

				if crops_num % 2 != 0:
					i += 1
					if i == crops_num - 2:
						break
				else:
					i += 2


if __name__ == '__main__':
	input_dir = '../../clic_data/images'
	out_dir = '../../clic_data/scm'

	os.makedirs(out_dir, exist_ok=True)

	for img_name in os.listdir(input_dir):
		image_path = os.path.join(input_dir, img_name)
		if os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
			print(f"Processing image: {image_path}")
			scm(image_path, 3, out_dir)
