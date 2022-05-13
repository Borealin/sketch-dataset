# encoding: utf-8
import glob
import logging
from functools import lru_cache
from os import path
from queue import Queue
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from sewar.full_ref import mse
from tqdm import tqdm

from utils import get_png_size, ProfileLoggingThread

Image.MAX_IMAGE_PIXELS = None
MIN_WIDTH = 200
MIN_HEIGHT = 2000
MAX_WIDTH = 200
MAX_HEIGHT = 20000


@lru_cache(maxsize=128)
def get_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path))


def compare_image(image1: np.ndarray, image2: np.ndarray) -> bool:
    x = min(image1.shape[0], image2.shape[0])
    y = min(image1.shape[1], image2.shape[1])
    c = min(image1.shape[2], image2.shape[2])

    res = mse(image1[:x, :y, :c], image2[:x, :y, :c])
    return res < 500


class MergeArtboardGroupThread(ProfileLoggingThread):
    def __init__(
            self,
            input_artboard_list_queue: Queue[List[str]],
            output_groups_queue: Queue[List[str]],
            thread_name: str,
            logging_file: str,
            profile_file: str,
            pbar: tqdm
    ):
        super().__init__(
            thread_name,
            logging_file,
            profile_file
        )
        self.input_artboard_list_queue = input_artboard_list_queue
        self.output_groups_queue = output_groups_queue
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.input_artboard_list_queue.empty():
                break
            try:
                artboard_list = self.input_artboard_list_queue.get()
                sub_image_groups = []
                self.logger.info(f'start processing {len(artboard_list)} artboards')
                for artboard in artboard_list:
                    image = get_image(artboard)
                    match = False
                    compare_count = 0
                    for group in sub_image_groups:
                        compare_count += 1
                        if compare_image(image, get_image(group[0])):
                            group.append(artboard)
                            match = True
                            break
                    if not match:
                        sub_image_groups.append([artboard])
                    self.logger.info(f'processed {artboard}, compared {compare_count} images')
                    self.pbar.update(1)
                for group in sub_image_groups:
                    self.output_groups_queue.put(group)
            except Exception as e:
                self.logger.error(e)
            finally:
                self.input_artboard_list_queue.task_done()


def merge_artboard_group(
        sketch_folders: List[str],
        logging_folder: str,
        profile_folder: str,
        max_thread=8
) -> List[List[str]]:
    logging.info(f"{len(sketch_folders)} sketches found")
    artboards: List[Tuple[str, int, int]] = []
    for sketch_folder in tqdm(sketch_folders):
        for artboard_path in glob.glob(f"{sketch_folder}/*"):
            image_size = get_png_size(artboard_path)
            artboards.append((artboard_path, image_size[0], image_size[1]))
    logging.info(f"{len(artboards)} artboards found")
    size_groups: Dict[Tuple[int, int], List[str]] = {}
    for artboard_path, width, height in artboards:
        size_groups.setdefault((width, height), []).append(artboard_path)
    logging.info(f"{len(size_groups)} size groups found")
    size_groups = {
        key: value
        for key, value in size_groups.items()
        if len(value) > 1
    }
    logging.info(f"{len(size_groups)} size groups with more than 1 artboard found")
    size_groups = {
        key: value
        for key, value in size_groups.items()
        if MIN_WIDTH < key[0] <= MAX_WIDTH and MIN_HEIGHT < key[1] <= MAX_WIDTH
    }
    logging.info(f"{len(size_groups)} size groups with width between 100 and 1000 found")
    image_group_queue = Queue()
    input_artboard_queue = Queue()
    pbar = tqdm(total=sum([len(x) for x in size_groups.values()]))
    for size, image_paths in size_groups.items():
        input_artboard_queue.put(image_paths)
    for i in range(max_thread):
        thread_name = f"simgroup-thread-{i}"
        thread = MergeArtboardGroupThread(
            input_artboard_queue,
            image_group_queue,
            thread_name,
            path.join(logging_folder, f"{thread_name}.log"),
            path.join(profile_folder, f"{thread_name}.profile"),
            pbar
        )
        thread.daemon = True
        thread.start()
    input_artboard_queue.join()
    return [image_group for image_group in image_group_queue.queue if len(image_group) > 1]


def visualize_groups(image_groups: List[List[str]], output_folder: str):
    for group_idx, group in enumerate(tqdm(image_groups)):
        group_images = [Image.open(x) for x in group]
        new_image = Image.new(
            mode='RGB',
            size=(group_images[0].size[0] * len(group_images), group_images[0].size[1])
        )
        for i, image in enumerate(group_images):
            new_image.paste(image, (i * image.size[0], 0))
        new_image.save(f"{output_folder}/{group_idx}.png")
