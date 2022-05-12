import re
from os import path
from typing import List, Union, Optional, TypeVar, Dict, Generic
from dataclasses import dataclass, field
from fastclasses_json import dataclass_json, JSONMixin
from .sketchtool import SketchTool, ExportFormat, BackgroundColor

WrapperResultValue = TypeVar("WrapperResultValue")


@dataclass
class WrapperResult(Generic[WrapperResultValue]):
    value: WrapperResultValue
    stdout: str
    stderr: Optional[str] = None


@dataclass_json
@dataclass
class BBox(JSONMixin):
    x: float
    y: float
    width: float
    height: float

    @classmethod
    def from_float_list(cls, float_list: List[float]) -> "BBox":
        return cls(float_list[0], float_list[1], float_list[2], float_list[3])

    def to_float_list(self) -> List[float]:
        return [self.x, self.y, self.width, self.height]

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "BBox":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "BBox":
        pass


@dataclass_json
@dataclass
class ListLayer(JSONMixin):
    id: str
    name: str
    layers: List["ListLayer"]
    rect: BBox
    trimmed: BBox
    relative: BBox
    influence: BBox

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "ListLayer":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ListLayer":
        pass


@dataclass_json
@dataclass
class ListLayersPage(JSONMixin):
    id: str
    name: str
    bounds: BBox = field(
        metadata={
            "fastclasses_json": {
                "decoder": lambda x: BBox.from_float_list([float(y) for y in x.split(",")]),
                "encoder": lambda x: ",".join(str(y) for y in x.to_float_list()),
            }
        }
    )
    layers: List[ListLayer]

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "ListLayersPage":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ListLayersPage":
        pass


@dataclass_json
@dataclass
class ListLayersResult(JSONMixin):
    pages: List[ListLayersPage]

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "ListLayersResult":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ListLayersResult":
        pass


@dataclass_json
@dataclass
class ListArtboard(JSONMixin):
    id: str
    name: str
    rect: BBox
    trimmed: BBox

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "ListArtboard":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ListArtboard":
        pass


@dataclass_json
@dataclass
class ListArtboardsPage(JSONMixin):
    id: str
    name: str
    bounds: BBox = field(
        metadata={
            "fastclasses_json": {
                "decoder": lambda x: BBox.from_float_list([float(y) for y in x.split(",")]),
                "encoder": lambda x: ",".join(str(y) for y in x.to_float_list()),
            }
        }
    )
    artboards: List[ListArtboard]

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "ListArtboardsPage":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ListArtboardsPage":
        pass


@dataclass_json
@dataclass
class ListArtboardsResult(JSONMixin):
    pages: List[ListArtboardsPage]

    @classmethod
    def from_dict(cls, o: dict, *, infer_missing=True) -> "ListArtboardsResult":
        pass

    @classmethod
    def from_json(cls, json_data: Union[str, bytes], *, infer_missing=True) -> "ListArtboardsResult":
        pass


class SketchToolWrapper:
    def __init__(self, sketch_path: str):
        self.sketchtool = SketchTool(sketch_path)
        self.export = self.Export(self.sketchtool)
        self.list = self.List(self.sketchtool)

    class Export:
        def __init__(self, sketchtool: SketchTool):
            self.sketchtool = sketchtool

        @classmethod
        def handle_stdio(
                cls,
                output: str,
                items: List[str],
                formats: List[ExportFormat],
                stdout: bytes,
                stderr: bytes,
        ) -> WrapperResult[Dict[str, List[bool]]]:
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            lines = stdout.strip().split("\n")
            item_map_format = {
                item: [None] * len(formats)
                for item in items
            }
            for line in lines:
                if line:
                    match = re.match(
                        r"Exported (.*)\.(.*)", line
                    )
                    if match:
                        matched_item = match.group(1)
                        matched_format = match.group(2)
                        if matched_item in item_map_format:
                            try:
                                item_map_format[matched_item][formats.index(
                                    ExportFormat(matched_format)
                                )] = path.join(output, f"{matched_item}.{matched_format}")
                            except ValueError:
                                pass
            res = WrapperResult(
                value=item_map_format,
                stdout=stdout,
            )
            if stderr:
                res.stderr = stderr
            return res

        async def layers(
                self,
                document: str,
                output: str = "./",
                formats: List[ExportFormat] = (ExportFormat.PNG,),
                items: List[str] = (),
                scales: List[Union[int, float]] = (1,),
                save_for_web: bool = False,
                overwriting: bool = False,
                trimmed: bool = False,
                group_contents_only: Optional[bool] = None,
                background: Optional[BackgroundColor] = None,
                suffixing: Optional[bool] = None,
        ) -> WrapperResult[Dict[str, List[bool]]]:
            process = await self.sketchtool.export.layers(
                document=document,
                output=output,
                formats=formats,
                items=items,
                scales=scales,
                save_for_web=save_for_web,
                overwriting=overwriting,
                trimmed=trimmed,
                background=background,
                group_contents_only=group_contents_only,
                use_id_for_name=True,
                suffixing=suffixing,
            )
            stdout, stderr = await process.communicate()
            return SketchToolWrapper.Export.handle_stdio(
                output=output,
                items=items,
                formats=formats,
                stdout=stdout,
                stderr=stderr,
            )

        async def artboards(
                self,
                document: str,
                output: str = "./",
                formats: List[ExportFormat] = (ExportFormat.PNG,),
                items: List[str] = (),
                scales: List[Union[int, float]] = (1,),
                save_for_web: bool = False,
                overwriting: bool = False,
                trimmed: bool = False,
                group_contents_only: Optional[bool] = None,
                include_symbols: Optional[bool] = None,
                export_page_as_fallback: Optional[bool] = None,
                serial: Optional[bool] = None
        ) -> WrapperResult[Dict[str, List[bool]]]:
            process = await self.sketchtool.export.artboards(
                document=document,
                output=output,
                formats=formats,
                items=items,
                scales=scales,
                save_for_web=save_for_web,
                overwriting=overwriting,
                trimmed=trimmed,
                group_contents_only=group_contents_only,
                include_symbols=include_symbols,
                export_page_as_fallback=export_page_as_fallback,
                serial=serial,
            )
            stdout, stderr = await process.communicate()
            return SketchToolWrapper.Export.handle_stdio(
                output=output,
                items=items,
                formats=formats,
                stdout=stdout,
                stderr=stderr,
            )

    class List:
        def __init__(self, sketchtool: SketchTool):
            self.sketchtool = sketchtool

        async def layers(self, document: str) -> WrapperResult[ListLayersResult]:
            process = await self.sketchtool.list.layers(document)
            stdout, stderr = await process.communicate()
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            try:
                list_layer_result = ListLayersResult.from_json(stdout)
            except ValueError:
                list_layer_result = ListLayersResult([])
            result = WrapperResult(
                value=list_layer_result,
                stdout=stdout
            )
            if stderr:
                result.stderr = stderr
            return result

        async def artboards(self, document: str) -> WrapperResult[ListArtboardsResult]:
            process = await self.sketchtool.list.artboards(document)
            stdout, stderr = await process.communicate()
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            try:
                list_artboards_result = ListArtboardsResult.from_json(stdout)
            except ValueError:
                list_artboards_result = ListArtboardsResult([])
            result = WrapperResult(
                value=list_artboards_result,
                stdout=stdout
            )
            if stderr:
                result.stderr = stderr
            return result
