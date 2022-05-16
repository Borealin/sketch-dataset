import asyncio
import logging
from dataclasses import dataclass
from os import path, makedirs
from pathlib import Path
from queue import Queue
from shutil import move
from typing import List, Dict, Optional, Any

from PIL import Image
from fastclasses_json import dataclass_json, JSONMixin
from sketch_document_py.sketch_file import from_file, to_file
from tqdm import tqdm

from sketch_dataset.datasets.extend_sketch_file_format import ExtendArtboard
from sketch_dataset.sketchtool import SketchToolWrapper, DEFAULT_SKETCH_PATH, ExportFormat, ListLayer
from sketch_dataset.utils import extract_artboards_from_sketch, ProfileLoggingThread

ITEM_THRESHOLD = 1000


@dataclass_json
@dataclass
class ConvertedSketchConfig(JSONMixin):
    sketch_json: str
    sketch_file: str
    artboard_json: str
    artboard_image: str
    config_file: str
    output_dir: str


@dataclass
class ArtboardData:
    artboard_folder: 'Path'
    list_layer: 'ListLayer'
    main_image: 'Path'
    layer_images: List['Path']


@dataclass
class SketchData:
    sketch_folder: 'Path'
    sketch_path: 'Path'
    artboards: List['ArtboardData']


@dataclass
class Dataset:
    config: 'ConvertedSketchConfig'
    sketches: List['SketchData']

    @classmethod
    def from_config(cls, config_path: str):
        config = ConvertedSketchConfig.from_json(open(config_path).read())


def recursive_merge(dict1: Dict[str, Any], dict2: Dict[str, Any], keys: List[str]):
    result = {**dict1, **dict2}
    for key in keys:
        if key in dict1 and key in dict2:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                result[key] = recursive_merge(dict1[key], dict2[key], keys)
    return result


async def convert_sketch(
        sketch_path: str,
        convert_config: ConvertedSketchConfig,
        logger: logging.Logger = None
):
    """
    Convert a sketch file to dataset.

    e.g.
    a sketch contains 3 artboards:
    1. artboard_1 (id: id_1) contains 1 layers (id: id_1_1, )
    2. artboard_2 (id: id_2) contains 2 layers (id: id_2_1, id: id_2_2)
    3. artboard_3 (id: id_3) contains 3 layers (id: id_3_1, id: id_3_2, id: id_3_3)

    resulted file tree will be:
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
            sketch_json_name.json
            sketch_file_name.sketch

    """
    sketch_output_folder = path.join(convert_config.output_dir, Path(sketch_path).stem)
    path.isdir(sketch_output_folder) or makedirs(sketch_output_folder, exist_ok=True)
    shrink_sketch_path = path.join(sketch_output_folder, convert_config.sketch_name)

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
    first_level_layer_dict: Dict[str, ListLayer] = {
        layer.id: layer
        for page in list_layers_coroutine_res.value.pages
        for layer in page.layers
    }

    # export artboard main image
    artboard_dict: Dict[str, ExtendArtboard] = {
        artboard.do_objectID: ExtendArtboard.from_dict(
            recursive_merge(
                artboard.to_dict(),
                first_level_layer_dict[artboard.do_objectID].to_dict(),
                ['layers']
            )
        )
        for artboard in extract_artboards_from_sketch(sketch_file)
    }
    export_format = ExportFormat.PNG
    export_artboards_coroutine = sketchtool.export.artboards(
        sketch_path,
        output=sketch_output_folder,
        formats=[export_format],
        items=artboard_dict.keys()
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
            artboard_json = first_level_layer_dict[artboard_id]

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
            artboard_json_path = path.join(artboard_output_folder, convert_config.artboard_json)
            with open(artboard_json_path, "w") as layer_json:
                layer_json.write(artboard_json.to_json())

            # write artboard image
            artboard_image_path = path.join(artboard_output_folder, convert_config.artboard_image)
            move(path.join(sketch_output_folder, f"{artboard_id}.{export_format.value}"), artboard_image_path)

    for export_layers_coroutine in export_layers_coroutines:
        export_layers_coroutine_res = await export_layers_coroutine
        if export_layers_coroutine_res.stderr and logger:
            logger.error(f'convert {sketch_path} failed when export layers: ')
            logger.error(export_layers_coroutine_res.stderr)


def convert_sketch_sync(
        sketch_path: str,
        convert_config: ConvertedSketchConfig,
        logger: logging.Logger = None
):
    return asyncio.run(
        convert_sketch(sketch_path, convert_config, logger))


class ConvertSketchThread(ProfileLoggingThread):
    def __init__(
            self,
            sketch_queue: Queue[str],
            convert_config: ConvertedSketchConfig,
            thread_name: str,
            logfile_path: str,
            profile_path: str,
            pbar: tqdm,
    ):
        super().__init__(thread_name, logfile_path, profile_path)
        self.sketch_queue = sketch_queue
        self.convert_config = convert_config
        self.pbar = pbar

    def run_impl(self):
        while True:
            if self.sketch_queue.empty():
                break
            sketch_path = self.sketch_queue.get()
            try:
                convert_sketch_sync(
                    sketch_path,
                    self.convert_config,
                    self.logger
                )
            except Exception as e:
                self.logger.error(f"{sketch_path} {e}")
            finally:
                self.pbar.update()
                self.sketch_queue.task_done()


def convert(
        sketch_list: List[str],
        convert_config: ConvertedSketchConfig,
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
            convert_config,
            thread_name,
            path.join(logfile_folder, f"{thread_name}.log"),
            path.join(profile_folder, f"{thread_name}.profile"),
            pbar
        )
        thread.daemon = True
        thread.start()
    sketch_queue.join()
    with open(path.join(convert_config.output_dir, convert_config.config_file), "w") as f:
        f.write(convert_config.to_json())


def draw_artboard(
        artboard_folder: str,
        artboard_json_name: str,
        artboard_export_image_name: Optional[str] = None,
        output_image_path: Optional[str] = None,
):
    artboard = ListLayer.from_json(open(path.join(artboard_folder, artboard_json_name), 'r').read())

    def dfs(layer: ListLayer, root: ListLayer, canvas: Image.Image) -> Image.Image:
        if len(layer.layers) > 0:
            for child in layer.layers:
                dfs(child, root, canvas)
        else:
            layer_image_path = path.join(artboard_folder, f"{layer.id}.png")
            if path.exists(layer_image_path):
                image = Image.open(layer_image_path).convert("RGBA")
                canvas.alpha_composite(image, (
                    int(layer.trimmed.x - artboard.trimmed.x), int(layer.trimmed.y - artboard.trimmed.y)))
        return canvas

    res = dfs(artboard, artboard,
              Image.new("RGBA", (int(artboard.rect.width), int(artboard.rect.height)), (255, 255, 255, 255)))
    if output_image_path is not None:
        res.save(output_image_path)
    if artboard_export_image_name is not None:
        real_res = Image.open(path.join(artboard_folder, artboard_export_image_name)).convert("RGBA")
        compare = Image.new("RGBA", (res.width * 2, res.height), (255, 255, 255, 255))
        compare.paste(res, (0, 0))
        compare.paste(real_res, (res.width, 0))
        compare.show()


__all__ = ['convert', 'draw_artboard', 'ConvertedSketchConfig']
