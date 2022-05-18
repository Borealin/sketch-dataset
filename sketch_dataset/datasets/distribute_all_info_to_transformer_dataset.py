import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Callable, Any

from PIL import Image
from fastclasses_json import dataclass_json
from sketch_document_py import sketch_file_format as sff
from tqdm.contrib.concurrent import thread_map

from sketch_dataset.datasets import Dataset, ArtboardData
from sketch_dataset.sketchtool.sketchtool_wrapper import ListLayer, BaseListLayer
from sketch_dataset.utils import get_png_size


@dataclass_json
@dataclass
class ExtendForReadListLayer(ListLayer):
    """
    Extend ListLayer with class and style for reading part from json
    """
    class_: str = field(metadata={'fastclasses_json': {'field_name': '_class'}})
    layers: List['ExtendForReadListLayer']
    style: Optional['sff.Style'] = None

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ExtendForReadListLayer":
        pass


@dataclass_json
@dataclass
class ExtendForWriteListLayer(BaseListLayer):
    """
    Extend ListLayer with class, color and label for writing to json
    """
    class_: str = field(metadata={'fastclasses_json': {'field_name': '_class'}})
    layers: List['ExtendForWriteListLayer']
    color: Tuple[float, float, float, float] = None
    label: int = 0

    @classmethod
    def from_for_read_list_layer(
            cls,
            list_layer: ExtendForReadListLayer
    ) -> "ExtendForWriteListLayer":
        return cls(
            name=list_layer.name,
            id=list_layer.id,
            rect=list_layer.trimmed,
            class_=list_layer.class_,
            layers=[
                cls.from_for_read_list_layer(layer)
                for layer in list_layer.layers
            ],
            color=extract_color_from_style(list_layer.style),
        )

    def flatten(self) -> List["ExtendForWriteListLayer"]:
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
    """
    Convert name to label
    @param name: layer name
    @return: label int
    """
    return 1 if name.startswith("#merge#") else 0


def draw_artboard_with_label(artboard: ArtboardData[ExtendForWriteListLayer]):
    artboard_root = artboard.list_layer

    def dfs(layer: ExtendForWriteListLayer, root: ExtendForWriteListLayer,
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


def default_artboard_filter(artboard: ArtboardData[ExtendForWriteListLayer]) -> bool:
    width_match = 200 < get_png_size(artboard.main_image)[0] < 2000
    height_match = 400 < get_png_size(artboard.main_image)[1] < 4000
    seq_len_match = 20 < len(artboard.list_layer.flatten()) < 200
    return width_match and height_match and seq_len_match


def layer_bbox_transform(
        layer: ExtendForWriteListLayer,
        root: ExtendForWriteListLayer
) -> ExtendForWriteListLayer:
    layer.rect.x = layer.rect.x - root.rect.x
    layer.rect.y = layer.rect.y - root.rect.y
    return layer


def convert_from_config(
        config_path: str,
        output_path: str,
        assets_image_size: Tuple[int, int],
        artboard_filter: Callable[[ArtboardData[ExtendForReadListLayer]], bool] = default_artboard_filter
) -> None:
    output_path = Path(output_path)
    dataset: Dataset[ExtendForReadListLayer] = Dataset.from_config(config_path, ExtendForReadListLayer)
    flatten_artboards = [
        artboard
        for sketch in dataset.sketches
        for artboard in sketch.artboards
    ]
    print(f"Found {len(flatten_artboards)} artboards")
    flatten_artboards = [
        artboard
        for artboard in flatten_artboards
        if artboard_filter(artboard)
    ]
    print(f"Remain {len(flatten_artboards)} artboards after filtered")

    def convert_artboard(input_data: Tuple[int, ArtboardData[ExtendForReadListLayer]]) -> Dict[str, str]:
        index, artboard = input_data
        for_read_root = artboard.list_layer
        for_write_root: ExtendForWriteListLayer = ExtendForWriteListLayer.from_for_read_list_layer(for_read_root)

        @dataclass
        class Context:
            label: int
            first: bool

        def update_label(list_layer: ExtendForWriteListLayer, context: Context) -> None:
            if len(list_layer.layers) > 0:
                label = name_to_label(list_layer.name)
                context = Context(label, True)
                for nest_layer in list_layer.layers:
                    update_label(nest_layer, context)
            else:
                if context.first:
                    list_layer.label = context.label * 2
                    context.first = False
                else:
                    list_layer.label = context.label * 2 + 1

        update_label(for_write_root, Context(0, True))

        # new_artboard: ArtboardData[ExtendClassAndColorListLayer] = copy.copy(artboard)
        # new_artboard.list_layer = for_write_root
        # draw_artboard_with_label(new_artboard)

        flatten_layers = [layer_bbox_transform(layer, for_write_root) for layer in for_write_root.flatten()]
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

        def layer_to_dict(layer: ExtendForWriteListLayer) -> Dict[str, Any]:
            layer_dict = layer.to_dict()
            del layer_dict["layers"]
            return layer_dict

        json.dump({
            "layers": [layer_to_dict(layer) for layer in flatten_layers],
            "width": for_write_root.rect.width,
            "height": for_write_root.rect.height,
            "layer_width": assets_image_size[0],
            "layer_height": assets_image_size[1]
        }, open(output_path.joinpath(json_name), "w"))
        Image.open(artboard.main_image).convert("RGBA").save(output_path.joinpath(image_name))
        assest_image.save(output_path.joinpath(layerassets))
        return {
            "json": json_name,
            "image": image_name,
            "layerassets": layerassets,
        }

    index_lst = thread_map(convert_artboard, list(enumerate(flatten_artboards)), max_workers=16)
    json.dump(index_lst, open(output_path.joinpath("index.json"), "w"))
