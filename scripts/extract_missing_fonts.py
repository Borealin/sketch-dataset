import glob
from os import path
import argparse
from utils import get_missing_font

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract missing fonts from a logging folder')
    parser.add_argument('--folder', type=str, help='Folder of logfile to extract missing fonts from')
    args = parser.parse_args()
    font_set = set()
    for log in glob.glob(path.join(args.folder, "*.log")):
        font_set.update(get_missing_font(log))
    print("\n".join(sorted(font_set)))
