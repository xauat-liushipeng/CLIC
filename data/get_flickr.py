# !/usr/bin/env python
# -*- coding: UTF-8 -*-
# ==================================================
# @Proj -> File
#         ：CLIC -> get_flickr
# @IDE    ：PyCharm
# @Date   ：2024/11/16 09:38
# @info   ：download and parse Flickr dataset
# ==================================================
import json
import os
import sys

import pandas as pd
import requests
from tqdm import tqdm


def download_parquet_urls(url, output_file="./parquet_urls.json"):
    # get request
    response = requests.get(url, stream=True)

    # check status code
    if response.status_code == 200:
        # save file to local
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"save to: {output_file}")
        return output_file
    else:
        print(f"download false, code: : {response.status_code}")
        sys.exit(-1)

def download_parquets(parquet_urls_file, output_dir="../../Flickr/parquet/"):
    # check path
    os.makedirs(output_dir, exist_ok=True)

    f = open(parquet_urls_file, "r")
    fli = json.load(f)

    # loop
    for part_url in fli:
        print('downloading {}'.format(part_url))

        # convert bytes to str
        if isinstance(part_url, bytes):
            part_url = part_url.decode('utf-8')

        file_name = part_url.split("/")[-1]
        output_path = os.path.join(output_dir, file_name)

        # get request
        response = requests.get(part_url, stream=True)
        if response.status_code == 200:
            # save
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded and saved: {output_path}")
        else:
            print(f"Failed to download {part_url}, status code: {response.status_code}")


def download_parquet_images(parquet_path, image_size='url_m', save_root='../../Flickr/train/'):
    print('downloading from {}'.format(parquet_path))
    # read parquet
    df = pd.read_parquet(parquet_path)

    # set image size in parquet, default url_m in our experiment
    url_column = image_size

    # mkdir for saving images
    output_dir = save_root + os.path.basename(parquet_path).split(".")[0]
    os.makedirs(output_dir, exist_ok=True)

    # loop for download images
    for index, row in tqdm(df.iterrows()):
        image_url = row.get(url_column)
        image_id = row.get('id')
        if pd.notna(image_url):
            try:
                # get
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    # save
                    file_extension = image_url.split('.')[-1]
                    file_path = os.path.join(output_dir, f"{image_id}.{file_extension}")
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    # print(f"Downloaded: {file_path}")
                else:
                    print(f"Failed to download {image_url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {image_url}: {e}")

if __name__ == '__main__':
    # Hugging Face API URL
    url = "https://huggingface.co/api/datasets/bigdata-pw/Flickr/parquet/default/train"

    parquet_urls_file = "./parquet_urls.json"
    parquet_save_dir = '../../Flickr/parquet/'

    image_size = 'url_m'
    images_save_dir = '../../Flickr/train/'

    # note: parquet_urls.json have been uploaded in ./data
    # urls_file = download_parquet_urls(url, parquet_urls_file)

    download_parquets(parquet_urls_file, parquet_save_dir)

    for parquet in os.listdir(parquet_save_dir):
        download_parquet_images(parquet_save_dir + parquet, image_size, images_save_dir)