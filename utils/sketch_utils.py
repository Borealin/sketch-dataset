from typing import List

from sketch_document_py import sketch_file as sf
from sketch_document_py import sketch_file_format as sff


def flatten_sketch_group(sketch_group: sff.AnyGroup) -> List[sff.AnyLayer]:
    """
    Flatten a SketchGroup object into a list of Sketch objects.
    """
    layers = []
    group_layers = sketch_group.layers
    sketch_group.layers = []
    layers.append(sketch_group)
    for layer in group_layers:
        if isinstance(layer, sff.AnyGroup.__args__):
            layers.extend(flatten_sketch_group(layer))
        else:
            layers.append(layer)
    return layers


def extract_artboards_from_sketch(sketch_file: sf.SketchFile) -> List[sff.Artboard]:
    """
    Flatten a SketchGroup object into a list of Sketch objects.
    """
    pages = sketch_file.contents.document.pages
    return [
        layer
        for page in pages
        for layer in page.layers
        if isinstance(layer, sff.Artboard)
    ]
