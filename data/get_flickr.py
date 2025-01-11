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
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pandas as pd
import requests
from conda.common.io import as_completed
from requests import HTTPError
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

    exist_files = os.listdir(output_dir)

    # loop for download images
    for index, row in df.iterrows():
        image_url = row.get(url_column)
        image_id = row.get('id')
        if image_id+'.jpg' in exist_files:
            print(image_id + 'exist!')
            continue
        print("image id: {}".format(image_id))
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
                    print(f"Downloaded: {file_path}")
                else:
                    print(f"Failed to download {image_url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {image_url}: {e}")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_image(image_url, image_id, output_dir):
    if pd.notna(image_url):
        try:
            # Send GET request with timeout
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                # Extract file extension and validate it
                file_extension = image_url.split('.')[-1].lower()
                if file_extension not in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:  # Common image formats
                    file_extension = 'jpg'

                # Save the image
                file_path = os.path.join(output_dir, f"{image_id}.{file_extension}")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Downloaded: {file_path}")
            else:
                logging.warning(f"Failed to download {image_url}, status code: {response.status_code}")
                if response.status_code == 429:
                    # If status code is 429, this thread will wait for 2 minutes
                    sleep(120)
                if response.status_code == 403:
                    sleep(300)
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error downloading {image_url}: {e}")
        except Exception as e:
            logging.error(f"Error downloading {image_url}: {e}")

def download_parquet_images2(parquet_path, image_size='url_m', save_root='../../Flickr/train/', max_workers=8):
    logging.info(f'Downloading from {parquet_path}')

    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    # Set the image size column
    url_column = image_size

    # Create output directory
    output_dir = os.path.join(save_root, os.path.basename(parquet_path).split(".")[0])
    os.makedirs(output_dir, exist_ok=True)

    # Get existing files
    exist_files = set(os.listdir(output_dir))

    # Prepare tasks for downloading images
    tasks = []
    for _, row in df.iterrows():
        image_url = row.get(url_column)
        image_id = row.get('id')
        if f"{image_id}.jpg" in exist_files or f"{image_id}.jpeg" in exist_files:
            logging.info(f"{image_id} exists! Skipping.")
            continue
        tasks.append((image_url, image_id, output_dir))

    # Use ThreadPoolExecutor to download images concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda args: download_image(*args), tasks), total=len(tasks), desc="Downloading images"))

    logging.info("Download completed.")


if __name__ == '__main__':
    # Hugging Face API URL
    # url = "https://huggingface.co/api/datasets/bigdata-pw/Flickr/parquet/default/train"

    # parquet_urls_file = "./parquet_urls.json"
    parquet_save_dir = '../../Flickr/'

    image_size = 'url_m'
    images_save_dir = '../../Flickr/train/'

    # note: parquet_urls.json have been uploaded in ./data
    # urls_file = download_parquet_urls(url, parquet_urls_file)

    # download_parquets(parquet_urls_file, parquet_save_dir)

    for parquet in os.listdir(parquet_save_dir):
        if (parquet.endswith('parquet')):
            download_parquet_images2(parquet_save_dir + parquet, image_size, images_save_dir)