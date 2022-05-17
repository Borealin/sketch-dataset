import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Callable

from PIL import Image
from fastclasses_json import dataclass_json
from sketch_document_py import sketch_file_format as sff
from tqdm.contrib.concurrent import thread_map

from sketch_dataset.datasets import Dataset, ArtboardData
from sketch_dataset.sketchtool.sketchtool_wrapper import ListLayer
from sketch_dataset.utils import get_png_size


@dataclass_json
@dataclass
class ExtendClassAndStyleListLayer(ListLayer):
    class_: str = field(metadata={'fastclasses_json': {'field_name': '_class'}})
    layers: List['ExtendClassAndStyleListLayer']
    style: Optional['sff.Style'] = None

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ExtendClassAndStyleListLayer":
        pass

    def flatten(self) -> List["ExtendClassAndStyleListLayer"]:
        return super().flatten()


@dataclass_json
@dataclass
class ExtendClassAndColorListLayer(ListLayer):
    class_: str = field(metadata={'fastclasses_json': {'field_name': '_class'}})
    layers: List['ExtendClassAndColorListLayer']
    style: Tuple[float, float, float, float] = None
    label: int = 0

    @classmethod
    def from_extend_class_and_style_list_layer(
            cls,
            list_layer: ExtendClassAndStyleListLayer
    ) -> "ExtendClassAndColorListLayer":
        return cls(
            id=list_layer.id,
            name=list_layer.name,
            rect=list_layer.rect,
            trimmed=list_layer.trimmed,
            relative=list_layer.relative,
            influence=list_layer.influence,
            class_=list_layer.class_,
            layers=[
                cls.from_extend_class_and_style_list_layer(layer)
                for layer in list_layer.layers
            ],
            style=extract_color_from_style(list_layer.style),
        )

    def flatten(self) -> List["ExtendClassAndColorListLayer"]:
        return super().flatten()


def extract_color_from_style(style: Optional['sff.Style']) -> Tuple[float, float, float, float]:
    if style is not None:
        fills = style.fills
        if fills is not None and len(fills) > 0:
            first_fill = fills[0]
            color = first_fill.color
            return color.red, color.green, color.blue, color.alpha

    return 0, 0, 0, 0


def name_to_label(name: str) -> int:
    return 1 if name.startswith("#merge#") else 0


def draw_artboard_with_label(artboard: ArtboardData[ExtendClassAndColorListLayer]):
    artboard_root = artboard.list_layer

    def dfs(layer: ExtendClassAndColorListLayer, root: ExtendClassAndColorListLayer,
            canvas: Image.Image) -> Image.Image:
        if len(layer.layers) > 0:
            for child in layer.layers:
                dfs(child, root, canvas)
        else:
            bbox = layer.trimmed
            bbox.x = bbox.x - root.trimmed.x
            bbox.y = bbox.y - root.trimmed.y
            image = Image.open(artboard.layer_images[layer.id]).convert("RGBA")
            canvas.alpha_composite(image, (
                int(bbox.x), int(bbox.y)))
            if layer.label == 1:
                canvas.alpha_composite(Image.new(canvas.mode, (int(bbox.width), int(bbox.height)), (255, 0, 0, 127)), (
                    int(bbox.x), int(bbox.y)))
        return canvas

    new_image = Image.new("RGBA", (int(artboard_root.rect.width), int(artboard_root.rect.height)), (255, 255, 255, 255))
    res = dfs(artboard_root, artboard_root, new_image)
    real_res = Image.open(artboard.main_image).convert("RGBA")
    compare = Image.new("RGBA", (res.width * 2, res.height), (255, 255, 255, 255))
    compare.paste(res, (0, 0))
    compare.paste(real_res, (res.width, 0))
    compare.show()


def convert_from_config(
        config_path: str,
        output_path: str,
        assets_image_size: Tuple[int, int],
        artboard_filter: Callable[[ArtboardData[ExtendClassAndStyleListLayer]], bool] = lambda x: 200 < get_png_size(
            x.main_image)[0] < 4000 and 200 < get_png_size(x.main_image)[1] < 8000
) -> None:
    output_path = Path(output_path)
    dataset: Dataset[ExtendClassAndStyleListLayer] = Dataset.from_config(config_path, ExtendClassAndStyleListLayer)
    flatten_artboards = [
        artboard
        for sketch in dataset.sketches
        for artboard in sketch.artboards
        if artboard_filter(artboard)
    ]

    def convert_artboard(input_data: Tuple[int, ArtboardData[ExtendClassAndStyleListLayer]]) -> Dict[str, str]:
        index, artboard = input_data
        old_list_layer = artboard.list_layer
        converted_list_layer: ExtendClassAndColorListLayer = ExtendClassAndColorListLayer.from_extend_class_and_style_list_layer(
            old_list_layer)

        def update_label(list_layer: ExtendClassAndColorListLayer, label: int):
            if len(list_layer.layers) > 0:
                label = name_to_label(list_layer.name)
                for nest_layer in list_layer.layers:
                    update_label(nest_layer, label)
            else:
                list_layer.label = label

        update_label(converted_list_layer, 0)

        # new_artboard: ArtboardData[ExtendClassAndColorListLayer] = copy.copy(artboard)
        # new_artboard.list_layer = converted_list_layer
        # draw_artboard_with_label(new_artboard)

        flatten_layers = converted_list_layer.flatten()
        json_name = f"{index}.json"
        image_name = f"{index}.png"
        layerassets = f"{index}-assets.png"

        assest_image = Image.new("RGBA", (assets_image_size[0], assets_image_size[1] * len(flatten_layers)),
                                 (255, 255, 255, 255))
        for index, layer in enumerate(flatten_layers):
            if layer.id not in artboard.layer_images:
                raise Exception(f"{layer.id} not in {artboard.layer_images}")
            assest_image.paste(
                Image.open(artboard.layer_images[layer.id]).convert("RGBA").resize(assets_image_size),
                (0, index * assets_image_size[1]))
        json.dump([layer.to_dict() for layer in flatten_layers], open(output_path.joinpath(json_name), "w"))
        Image.open(artboard.main_image).convert("RGBA").save(output_path.joinpath(image_name))
        assest_image.save(output_path.joinpath(layerassets))
        return {
            "json": json_name,
            "image": image_name,
            "layerassets": layerassets,
        }

    index_lst = thread_map(convert_artboard, list(enumerate(flatten_artboards)), max_workers=16)
    json.dump(index_lst, open(output_path.joinpath("index.json"), "w"))
