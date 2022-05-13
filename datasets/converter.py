import asyncio
import glob
import logging
from os import path, makedirs
from pathlib import Path
from queue import Queue
from shutil import move

from typing import List, Tuple, Dict

from sketch_document_py.sketch_file import from_file, to_file
from tqdm import tqdm

from sketchtool import SketchToolWrapper, DEFAULT_SKETCH_PATH, ExportFormat
from utils import extract_artboards_from_sketch, ProfileLoggingThread

ITEM_THRESHOLD = 1000
async def convert_sketch(
        sketch_path: str,
        output_folder: str,
        shrink_sketch_name: str,
        artboard_json_name: str,
        artboard_export_image_name: str,
        logger: logging.Logger = None
):
    """
    Convert a sketch file to dataset.

    e.g.
    a sketch contains 3 artboards:
    1. artboard_1 (id: id_1) contains 1 layers (id: id_1_1, )
    2. artboard_2 (id: id_2) contains 2 layers (id: id_2_1, id: id_2_2)
    3. artboard_3 (id: id_3) contains 3 layers (id: id_3_1, id: id_3_2, id: id_3_3)

    output_folder will be:
    output_folder/
        sketch_name/
            id_1/
                id_1_1.png
                artboard_json_name.json
                artboard_export_image_name.png
            id_2/
                id_2_1.png
                id_2_2.png
                artboard_json_name.json
                artboard_export_image_name.png
            id_3/
                id_3_1.png
                id_3_2.png
                id_3_3.png
                artboard_json_name.json
                artboard_export_image_name.png
            shrink_sketch_name.sketch

    """
    sketch_output_folder = path.join(output_folder, Path(sketch_path).stem)
    path.isdir(sketch_output_folder) or makedirs(sketch_output_folder, exist_ok=True)
    shrink_sketch_path = path.join(sketch_output_folder, shrink_sketch_name)

    # save shrunk sketch
    sketch_file = from_file(sketch_path)
    to_file(sketch_file, shrink_sketch_path)

    # init sketchtool
    sketchtool = SketchToolWrapper(DEFAULT_SKETCH_PATH)

    # read list layers
    list_layers_coroutine = sketchtool.list.layers(sketch_path)
    list_layers_coroutine_res = await list_layers_coroutine
    if list_layers_coroutine_res.stderr and logger:
        logger.error(f'convert {sketch_path} failed when list layers: ')
        logger.error(list_layers_coroutine_res.stderr)
    first_level_layers = [
        layer
        for page in list_layers_coroutine_res.value.pages
        for layer in page.layers
    ]

    # export artboard main image
    artboard_ids = [artboard.do_objectID for artboard in extract_artboards_from_sketch(sketch_file)]
    export_format = ExportFormat.PNG
    export_artboards_coroutine = sketchtool.export.artboards(
        sketch_path,
        output=sketch_output_folder,
        formats=[export_format],
        items=artboard_ids
    )
    export_artboards_coroutine_res = await export_artboards_coroutine
    if export_artboards_coroutine_res.stderr and logger:
        logger.error(f'convert {sketch_path} failed when export artboards: ')
        logger.error(export_artboards_coroutine_res.stderr)
    export_result: Dict[str, bool] = {k: v[0] for k, v in export_artboards_coroutine_res.value.items()}

    export_layers_coroutines = []
    for artboard_id, artboard_export_status in export_result.items():
        if artboard_export_status:
            # make artboard dir
            artboard_output_folder = path.join(sketch_output_folder, artboard_id)
            path.isdir(artboard_output_folder) or makedirs(artboard_output_folder, exist_ok=True)

            # get artboard json
            artboard_json = next(layer for layer in first_level_layers if layer.id == artboard_id)

            # get all layer id in artboard
            items = [layer.id for layer in artboard_json.flatten()]

            # export artboard layers
            for i in range(int(len(items) / ITEM_THRESHOLD) + 1):
                export_layers_coroutine = sketchtool.export.layers(
                    sketch_path,
                    output=artboard_output_folder,
                    formats=[export_format],
                    items=items[i * ITEM_THRESHOLD: (i + 1) * ITEM_THRESHOLD],
                )
                export_layers_coroutines.append(export_layers_coroutine)

            # write artboard json
            artboard_json_path = path.join(artboard_output_folder, artboard_json_name)
            with open(artboard_json_path, "w") as layer_json:
                layer_json.write(artboard_json.to_json())

            # write artboard image
            artboard_image_path = path.join(artboard_output_folder, artboard_export_image_name)
            move(path.join(sketch_output_folder, f"{artboard_id}.{export_format.value}"), artboard_image_path)

    for export_layers_coroutine in export_layers_coroutines:
        export_layers_coroutine_res = await export_layers_coroutine
        if export_layers_coroutine_res.stderr and logger:
            logger.error(f'convert {sketch_path} failed when export layers: ')
            logger.error(export_layers_coroutine_res.stderr)


def convert_sketch_sync(
        sketch_path: str,
        output_path: str,
        shrink_sketch_name: str,
        artboard_json_name: str,
        artboard_export_image_name: str,
        logger: logging.Logger = None
):
    return asyncio.run(
        convert_sketch(sketch_path, output_path, shrink_sketch_name, artboard_json_name, artboard_export_image_name, logger))


class ConvertSketchThread(ProfileLoggingThread):
    def __init__(
            self,
            sketch_queue: Queue[str],
            output_path: str,
            shrink_sketch_name: str,
            artboard_json_name: str,
            artboard_export_image_name: str,
            thread_name: str,
            logfile_path: str,
            profile_path: str,
            pbar: tqdm,
    ):
        super().__init__(thread_name, logfile_path, profile_path)
        self.sketch_queue = sketch_queue
        self.output_path = output_path
        self.shrink_sketch_name = shrink_sketch_name
        self.artboard_json_name = artboard_json_name
        self.artboard_export_image_name = artboard_export_image_name
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.sketch_queue.empty():
                break
            sketch_path = self.sketch_queue.get()
            try:
                convert_sketch_sync(
                    sketch_path,
                    self.output_path,
                    self.shrink_sketch_name,
                    self.artboard_json_name,
                    self.artboard_export_image_name,
                    self.logger
                )
            except Exception as e:
                self.logger.error(f"{sketch_path} {e}")
            finally:
                self.pbar.update()
                self.sketch_queue.task_done()


def process(
        sketch_list: List[str],
        output_folder: str,
        shrink_sketch_name: str,
        artboard_json_name: str,
        artboard_export_image_name: str,
        logfile_folder: str,
        profile_folder: str,
        max_threads: int = 4,
):
    pbar = tqdm(total=len(sketch_list))
    sketch_queue: Queue[str] = Queue()
    for sketch_path in sketch_list:
        sketch_queue.put(sketch_path)
    for i in range(max_threads):
        thread_name = f"convert-thread-{i}"
        thread = ConvertSketchThread(
            sketch_queue,
            output_folder,
            shrink_sketch_name,
            artboard_json_name,
            artboard_export_image_name,
            thread_name,
            path.join(logfile_folder, f"{thread_name}.log"),
            path.join(profile_folder, f"{thread_name}.profile"),
            pbar
        )
        thread.daemon = True
        thread.start()
    sketch_queue.join()


if __name__ == "__main__":
    output_path = "./converted"
    shrink_sketch_name = "main.sketch"
    artboard_json_name = "main.json"
    artboard_export_image_name = "main.png"
    process(
        glob.glob("/Users/bytedance/Documents/school/dataset/first_labeled/*.sketch"),
        output_path,
        shrink_sketch_name,
        artboard_json_name,
        artboard_export_image_name,
        8
    )
