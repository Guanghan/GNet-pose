import sys, os
sys.path.append(os.path.abspath("../utils/"))

import cv2
from utils_io_coord import *
from utils_io_list import *
from utils_io_folder import create_folder

from utils_human_segment import *


def test_segment_human():
    img_num = 1002
    img_fold = '/home/ngh/dev/POSE/POSE-dev/testing/my_baseline/dataset_lsp/images_all_cropped/qualitative/'
    img_path = os.path.join(img_fold, 'im' + str(img_num) + '.jpg')

    img = cv2.imread(img_path)
    img_segmented = segment_human(img)

    cv2.imshow('segmented image', img_segmented)
    cv2.waitKey(0)
    return True


def main():
    print("Testing: utils_human_segment")

    finished = test_segment_human()
    if finished is not True:
        print("test_segment_human failed")


if __name__ == '__main__':
    main()
