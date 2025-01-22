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
import threading
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pandas as pd
import requests
from conda.common.io import as_completed


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
                return True
            else:
                logging.warning(f"Failed to download {image_url}, status code: {response.status_code}")
                if response.status_code == 429:
                    # If status code is 429, this thread will wait for 2 minutes
                    sleep(120)
                else:
                    return False
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error downloading {image_url}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error downloading {image_url}: {e}")
            return False
    else:
        return False


def download_parquet_images(parquet_path, image_size='url_m', save_root='../../Flickr/train/', max_workers=8):
    # 初始化锁和计数器
    lock = threading.Lock()
    total_files_downloaded = [0]

    logging.info(f'Downloading from {parquet_path}')

    # Read the parquet file
    df = pd.read_parquet(parquet_path, columns=['id', 'url_m'])

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Set the image size column
    # url_column = image_size

    # Create output directory
    output_dir = os.path.join(save_root, os.path.basename(parquet_path).split(".")[0])
    os.makedirs(output_dir, exist_ok=True)

    # Get existing files
    exist_files = set(os.listdir(output_dir))

    # Prepare tasks for downloading images
    tasks = []
    for row in df.itertuples(index=True, name='Pandas'):
        image_url = row[2]  # image size column
        image_id = row[1]
        if f"{image_id}.jpg" in exist_files or f"{image_id}.jpeg" in exist_files:
            logging.info(f"{image_id} exists! Skipping.")
            continue
        tasks.append((image_url, image_id, output_dir))

    # thread safe
    def download_image_thread_safe(image_url, image_id, output_dir):
        nonlocal total_files_downloaded
        with lock:
            if total_files_downloaded[0] >= 300:
                return False

        # download
        if download_image(image_url, image_id, output_dir):
            with lock:
                total_files_downloaded[0] += 1
                if total_files_downloaded[0] >= 300:
                    return True

        return False

    # thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image_thread_safe, *task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                logging.info(f"Reached 300 images in {output_dir}. Shutting down.")
                executor.shutdown(wait=False)
                break

    logging.info("Download completed.")


def download_parquets(parquet_urls_file, output_dir="../../Flickr/parquet/"):
    # check path
    os.makedirs(output_dir, exist_ok=True)

    f = open(parquet_urls_file, "r")
    fli = json.load(f)

    # loop
    for part_url in fli:
        # convert bytes to str
        if isinstance(part_url, bytes):
            part_url = part_url.decode('utf-8')

        file_name = part_url.split("/")[-1]
        output_path = os.path.join(output_dir, file_name)

        if not os.path.exists(output_path):
            print('Downloading parquet: {}'.format(part_url))
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
        else:
            print('Existing parquet: {}'.format(part_url))

        download_parquet_images(output_path, image_size, images_save_dir)

        os.remove(output_path)

        print('Removed parquet: {}'.format(part_url))

if __name__ == '__main__':
    parquet_urls_file = "./parquet_urls.json"
    parquet_save_dir = '../../Flickr/'

    image_size = 'url_m'
    images_save_dir = '../../Flickr_train/'

    download_parquets(parquet_urls_file, parquet_save_dir)