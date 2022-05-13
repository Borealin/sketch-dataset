import glob
import json
import logging
from os import path

from datasets import merge_artboard_group, visualize_groups, export_artboards
from utils import create_folder

if __name__ == "__main__":
    logging.basicConfig(filename='../main.log', level=logging.INFO)

    input_sketch_folder = '../input/sketch'

    export_artboard_folder = "F:/dataset/first_unlabeled/sketch_artboards"
    export_artboard_logging_folder = path.join(export_artboard_folder, "logging")
    export_artboard_profile_folder = path.join(export_artboard_folder, "profile")

    similarity_res_folder = "sim_res"
    similarity_logging_folder = path.join(similarity_res_folder, "logging")
    similarity_profile_folder = path.join(similarity_res_folder, "profile")
    sim_groups_json = path.join(similarity_res_folder, "sim_groups.json")

    for folder in [export_artboard_folder, export_artboard_logging_folder, export_artboard_profile_folder,
                   similarity_res_folder, similarity_logging_folder, similarity_profile_folder]:
        create_folder(folder)

    export_artboards(
        glob.glob(f"{input_sketch_folder}/*.sketch"),
        export_artboard_folder,
        similarity_logging_folder,
        similarity_profile_folder,
        12
    )
    groups = merge_artboard_group(
        glob.glob(f"{export_artboard_folder}/*{path.sep}"),
        similarity_logging_folder,
        similarity_profile_folder,
        12
    )
    json.dump(
        groups,
        open(sim_groups_json, 'w')
    )
    groups = json.load(open(sim_groups_json, 'r'))
    visualize_groups(groups, similarity_res_folder)
