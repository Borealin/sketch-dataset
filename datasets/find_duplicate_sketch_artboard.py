# encoding: utf-8
import asyncio
import glob
import json
import re
from functools import lru_cache
from os import path, makedirs
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from sewar.full_ref import mse
from tqdm import tqdm

from sketchtool import SketchToolWrapper, WrapperResult, DEFAULT_SKETCH_PATH
from utils import get_png_size

from threading import Thread

Image.MAX_IMAGE_PIXELS = None


async def generate_artboards_png(sketch_path: str, output: str) -> WrapperResult[Dict[str, List[bool]]]:
    sketchtool = SketchToolWrapper(DEFAULT_SKETCH_PATH)
    artboards = [artboard for page in (await sketchtool.list.artboards(sketch_path)).value.pages for artboard in
                 page.artboards]
    output = path.join(output, Path(sketch_path).stem)
    path.isdir(output) or makedirs(output, exist_ok=True)
    return await sketchtool.export.artboards(
        sketch_path,
        output=output,
        items=[artboard.id for artboard in artboards],
        overwriting=True
    )


def generate_artboards_png_sync(sketch_path: str, output: str) -> WrapperResult[Dict[str, List[bool]]]:
    return asyncio.run(generate_artboards_png(sketch_path, output))


class GenerateArtboardsThread(Thread):
    def __init__(self, sketch_queue: Queue, output: str, pbar: tqdm, error_queue: Queue):
        super().__init__()
        self.sketch_queue = sketch_queue
        self.output = output
        self.pbar = pbar
        self.error_queue = error_queue

    def run(self):
        while True:
            if self.sketch_queue.empty():
                break
            sketch_path = self.sketch_queue.get()
            res = generate_artboards_png_sync(sketch_path, self.output)
            if res.stderr:
                self.error_queue.put(res.stderr)
            self.pbar.update()
            self.sketch_queue.task_done()


def process(sketch_list: List[str], output: str, error_file: str):
    pbar = tqdm(total=len(sketch_list))
    sketch_queue = Queue()
    error_queue = Queue()
    for sketch_path in sketch_list:
        sketch_queue.put(sketch_path)
    for _ in range(8):
        thread = GenerateArtboardsThread(sketch_queue, output, pbar, error_queue)
        thread.daemon = True
        thread.start()
    sketch_queue.join()
    error_message_set = set()
    while not error_queue.empty():
        stderr: str = error_queue.get()
        for line in stderr.strip().split("\n"):
            error_message_set.add(line + "\n")
    with open(error_file, "w") as f:
        f.writelines(list(error_message_set))


def get_missing_font(error_file: str) -> List[str]:
    missing_fonts = set()
    with open(error_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = re.match(r".+Client requested name \"(((?!\").)+)\"", line)
            if match:
                missing_fonts.add(match.group(1))
    return list(sorted(missing_fonts))


@lru_cache(maxsize=128)
def get_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path))


def compare_image(image1: np.ndarray, image2: np.ndarray) -> bool:
    x = min(image1.shape[0], image2.shape[0])
    y = min(image1.shape[1], image2.shape[1])
    c = min(image1.shape[2], image2.shape[2])

    res = mse(image1[:x, :y, :c], image2[:x, :y, :c])
    return res < 500


def merge_artboard_group(output_folder: str, max_thread=8) -> List[List[str]]:
    sketch_folders = glob.glob(f"{output_folder}/*{path.sep}")
    artboards: List[Tuple[str, int, int]] = []
    for sketch_folder in tqdm(sketch_folders):
        for artboard_path in glob.glob(f"{sketch_folder}/*"):
            image_size = get_png_size(artboard_path)
            artboards.append((artboard_path, image_size[0], image_size[1]))
    print(f"{len(artboards)} artboards found")
    size_groups: Dict[Tuple[int, int], List[str]] = {}
    for artboard_path, width, height in artboards:
        size_groups.setdefault((width, height), []).append(artboard_path)
    print(f"{len(size_groups)} size groups found")
    size_groups = {
        key: value
        for key, value in size_groups.items()
        if len(value) > 1
    }
    print(f"{len(size_groups)} size groups with more than 1 artboard found")
    size_groups = {
        key: value
        for key, value in size_groups.items()
        if 100 < key[0] and key[0] <= 1000 and 100 < key[1] and key[1] <= 10000
    }
    print(f"{len(size_groups)} size groups with width and height between 100 and 1000 found")
    image_group_queue = Queue()
    input_artboard_queue = Queue()
    pbar = tqdm(total=sum([len(x) for x in size_groups.values()]))
    for size, image_paths in size_groups.items():
        input_artboard_queue.put(image_paths)
    for _ in range(max_thread):
        thread = MergeArtboardGroupThread(image_group_queue, input_artboard_queue, pbar)
        thread.daemon = True
        thread.start()
    input_artboard_queue.join()
    image_groups: List[List[str]] = []
    while not image_group_queue.empty():
        image_groups.append(image_group_queue.get())
        image_group_queue.task_done()
    return image_groups


class MergeArtboardGroupThread(Thread):
    def __init__(self, image_groups_queue: Queue[List[str]], input_artboard_list_queue: Queue[List[str]], pbar: tqdm):
        super().__init__()
        self.image_groups_queue = image_groups_queue
        self.input_artboard_list_queue = input_artboard_list_queue
        self.pbar = pbar

    def run(self):
        while True:
            if self.input_artboard_list_queue.empty():
                break
            artboard_list = self.input_artboard_list_queue.get()
            sub_image_groups = []
            for artboard in artboard_list:
                image = get_image(artboard)
                match = False
                for group in sub_image_groups:
                    if compare_image(image, get_image(group[0])):
                        group.append(artboard)
                        match = True
                        break
                if not match:
                    sub_image_groups.append([artboard])
                self.pbar.update(1)
            for group in sub_image_groups:
                self.image_groups_queue.put(group)
            self.input_artboard_list_queue.task_done()


if __name__ == "__main__":
    output_folder = "F:\dataset\sim_output"
    sim_folder = "sim"
    sim_groups_json = path.join(sim_folder, "sim_groups.json")
    path.isdir(output_folder) or makedirs(output_folder, exist_ok=True)
    path.isdir(sim_folder) or makedirs(sim_folder, exist_ok=True)
    # process(
    #     glob.glob("/Users/bytedance/Documents/school/dataset/first_unlabeled/Alibaba*.sketch"),
    #     output_folder,
    #     "./error.txt"
    # )
    json.dump(
        list(
            filter(
                lambda x: len(x) > 1,
                merge_artboard_group(output_folder, 12)
            )
        ),
        open(sim_groups_json, 'w')
    )
    groups = json.load(open(sim_groups_json, 'r'))
    for group_idx, group in enumerate(tqdm(groups)):
        group_images = [Image.open(x) for x in group]
        new_image = Image.new(
            mode='RGB',
            size=(group_images[0].size[0] * len(group_images), group_images[0].size[1])
        )
        for i, image in enumerate(group_images):
            new_image.paste(image, (i * image.size[0], 0))
        new_image.save(f"{sim_folder}/{group_idx}.png")
