import argparse
import os


# Constants
TEST_IMAGE_NAME = "1.jpg"


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def dir_filter_criterion(path):
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, TEST_IMAGE_NAME)):
        return False
    return True


def handle_directory(path, start_ind):
    ind = 0 if start_ind == None else start_ind
    dir_contents = sorted(list(filter(lambda dirname: dir_filter_criterion(os.path.join(path, dirname)), os.listdir(path))))
    print(f"############# HANDLING A TOTAL OF {len(dir_contents)} DIRECTORIES #############")
    for i, dir in enumerate(dir_contents[start_ind:]):
        print(f"HANDLING #{i + ind}: {dir}")
        os.system(f"python data_labeling/specifier.py \"{os.path.join(path, dir, TEST_IMAGE_NAME)}\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Tool to loop over the files in TuSimple and run the labeling
                                                    specifier script, just to makes things a bit easier.""")
    parser.add_argument("dirname", type=dir_path, default=None, nargs=1)
    parser.add_argument("--start", metavar="start_dir_number", type=int, nargs='?',
                    help="Specifies the directory number (in the sorted list of directory names) to start the labeler from.")
    args = parser.parse_args()

    handle_directory(os.path.abspath(args.dirname[0]), args.start)