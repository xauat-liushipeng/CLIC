#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Proj -> File
		：clic -> clic -> rcm.py
@IDE    ：PyCharm
@Author ：liu shipeng
@Date   ：2024/11/12
@info   ：random crop and merge
=================================================='''
import random

from PIL import Image

def crop(width, height, img, size):
	if size > width or size > height:
		raise ValueError()

	# left top
	x = random.randint(0, width - size)
	y = random.randint(0, height - size)

	# crop
	cropped_img = img.crop((x, y, x + size, y + size))

	return cropped_img

# todo
def transforms():
	pass

# todo
def merge():
	pass

def random_crop(image_path, c):
	with Image.open(image_path) as img:
		width, height = img.size

		l_crops_num = c
		m_crops_num = 2 * c
		s_crops_num = 4 * c

		l_crops_size = int(width / l_crops_num)
		m_crops_size = int(width / m_crops_num)
		s_crops_size = int(width / s_crops_num)

		for i in range(l_crops_num):
			crop(width, height, img, l_crops_size).save(out_dir + 'l_' + str(i + 1) + '.jpg')
		for i in range(m_crops_num):
			crop(width, height, img, m_crops_size).save(out_dir + 'm_' + str(i + 1) + '.jpg')
		for i in range(s_crops_num):
			crop(width, height, img, s_crops_size).save(out_dir + 's_' + str(i + 1) + '.jpg')


if __name__ == '__main__':
	image_path = 'scenes_sky_00000629.jpg'
	out_dir = 'RCM_3/'
	random_crop(image_path, 3)