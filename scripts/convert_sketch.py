import argparse
import glob
from dataclasses import dataclass
from os import path

from datasets import convert
from utils import create_folder


def main(base_folder: str):
    output_path = path.join(base_folder, "converted")
    shrink_sketch_name = "main.sketch"
    artboard_json_name = "main.json"
    artboard_export_image_name = "main.png"
    logfile_folder = path.join(output_path, "logging")
    profile_folder = path.join(output_path, "profile")
    for folder in [output_path, logfile_folder, profile_folder]:
        create_folder(folder)
    convert(
        glob.glob(path.join(base_folder, "sketches/*.sketch")),
        output_path,
        shrink_sketch_name,
        artboard_json_name,
        artboard_export_image_name,
        logfile_folder,
        profile_folder,
        8
    )


@dataclass
class ConvertNameSpace:
    input: str = ""
    output: str = ""
    shrunk_sketch: str = ""
    artboard_json: str = ""
    artboard_image: str = ""
    logfile_folder: str = ""
    profile_folder: str = ""
    threads: int = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert sketches in folder to dataset format')
    parser.add_argument('--input', type=str, help='Folder containing sketches')
    parser.add_argument('--output', type=str, help='Output folder')
    parser.add_argument('--shrunk_sketch', type=str, help='Name of shrunk sketch file', default="main.sketch")
    parser.add_argument('--artboard_json', type=str, help='Name of artboard json file contains layers bbox structure',
                        default="main.json")
    parser.add_argument('--artboard_image', type=str, help='Name of exported artboard image file', default="main.png")
    parser.add_argument('--logfile_folder', type=str, help='Folder to save log files')
    parser.add_argument('--profile_folder', type=str, help='Folder to save profile files')
    parser.add_argument('--threads', type=int, help='Number of threads to use', default=8)
    args = parser.parse_args(namespace=ConvertNameSpace())

    if args.input is None:
        raise ValueError("Input folder is not specified")
    if args.logfile_folder is None:
        args.logfile_folder = path.join(args.output, "logging")
    if args.profile_folder is None:
        args.profile_folder = path.join(args.output, "profile")

    for folder in [args.output, args.logfile_folder, args.profile_folder]:
        create_folder(folder)
    convert(
        glob.glob(path.join(args.input, "*.sketch")),
        args.output,
        args.shrunk_sketch,
        args.artboard_json,
        args.artboard_image,
        args.logfile_folder,
        args.profile_folder,
        8
    )
