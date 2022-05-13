import asyncio
import glob
from os import path, makedirs
from pathlib import Path
from queue import Queue
from shutil import move
from threading import Thread
from typing import List, Tuple, Dict

from sketch_document_py.sketch_file import from_file, to_file
from tqdm import tqdm

from sketchtool import SketchToolWrapper, DEFAULT_SKETCH_PATH, ExportFormat
from utils import extract_artboards_from_sketch


async def convert_sketch(
        sketch_path: str,
        output_path: str,
        shrink_sketch_name: str,
        artboard_json_name: str,
        artboard_export_image_name: str,
):
    sketch_output_folder = path.join(output_path, Path(sketch_path).stem)
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
    export_result: Dict[str, bool] = {k: v[0] for k, v in export_artboards_coroutine_res.value.items()}

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
            for i in range(int(len(items) / 100) + 1):
                export_layers_coroutine = sketchtool.export.layers(
                    sketch_path,
                    output=artboard_output_folder,
                    formats=[export_format],
                    items=items[i * 100: (i + 1) * 100],
                )
                await export_layers_coroutine

            # write artboard json
            artboard_json_path = path.join(artboard_output_folder, artboard_json_name)
            with open(artboard_json_path, "w") as layer_json:
                layer_json.write(artboard_json.to_json())

            # write artboard image
            artboard_image_path = path.join(artboard_output_folder, artboard_export_image_name)
            move(path.join(sketch_output_folder, f"{artboard_id}.{export_format.value}"), artboard_image_path)


def convert_sketch_sync(
        sketch_path: str,
        output_path: str,
        shrink_sketch_name: str,
        artboard_json_name: str,
        artboard_export_image_name: str,
):
    return asyncio.run(
        convert_sketch(sketch_path, output_path, shrink_sketch_name, artboard_json_name, artboard_export_image_name))


class ConvertSketchThread(Thread):
    def __init__(
            self,
            sketch_queue: Queue[str],
            output_path: str,
            shrink_sketch_name: str,
            artboard_json_name: str,
            artboard_export_image_name: str,
            error_queue: Queue[Tuple[str, Exception]],
            pbar: tqdm,
    ):
        super().__init__()
        self.sketch_queue = sketch_queue
        self.output_path = output_path
        self.shrink_sketch_name = shrink_sketch_name
        self.artboard_json_name = artboard_json_name
        self.artboard_export_image_name = artboard_export_image_name
        self.error_queue = error_queue
        self.pbar = pbar

    def run(self):
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
                    self.artboard_export_image_name
                )
            except Exception as e:
                self.error_queue.put((sketch_path, e))
            finally:
                self.pbar.update()
                self.sketch_queue.task_done()


def process(
        sketch_list: List[str],
        output_path: str,
        shrink_sketch_name: str,
        artboard_json_name: str,
        artboard_export_image_name: str,
        error_message_file: str,
        max_threads: int = 4,
):
    pbar = tqdm(total=len(sketch_list))
    sketch_queue: Queue[str] = Queue()
    error_message_queue: Queue[Tuple[str, Exception]] = Queue()
    for sketch_path in sketch_list:
        sketch_queue.put(sketch_path)
    for _ in range(max_threads):
        thread = ConvertSketchThread(
            sketch_queue,
            output_path,
            shrink_sketch_name,
            artboard_json_name,
            artboard_export_image_name,
            error_message_queue,
            pbar
        )
        thread.daemon = True
        thread.start()
    sketch_queue.join()
    if not error_message_queue.empty():
        with open(error_message_file, "w") as f:
            for sketch_path, e in error_message_queue.queue:
                f.write(f"{sketch_path}: \n{e}\n")


if __name__ == "__main__":
    output_path = "./converted"
    shrink_sketch_name = "main.sketch"
    artboard_json_name = "main.json"
    artboard_export_image_name = "main.png"
    error_message_file = "error.txt"
    process(
        glob.glob("/Users/bytedance/Documents/school/dataset/first_labeled/*.sketch"),
        output_path,
        shrink_sketch_name,
        artboard_json_name,
        artboard_export_image_name,
        error_message_file,
        8
    )
