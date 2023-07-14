import os
import logging


def create_dirs(dirs: list):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)

def create_video(path: str):
    """
    path - a str indicating the path of the video to be created
    """
    try:
        video = open(path, 'wb')
        video.close()
    except:
        print(f"Could not create {path}")

    