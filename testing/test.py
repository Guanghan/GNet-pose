'''
 Batch Test Human Pose Estimation
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''
import caffe, cv2, Image, scipy.io, sys, os
import numpy as np

caffe_root = '/home/ngh/dev/Github/caffe-GNet/'
sys.path.insert(0, caffe_root + 'python')

sys.path.append(os.path.abspath("utils/"))
sys.path.append(os.path.abspath("testing/utils/"))
from utils_io_file import is_image
from utils_io_folder import *
from utils_convert_heatmap import *

from utils_pose import *
from utils_nms import find_joints_in_heatmaps_nms


def batch_test_images(folder_path_in, folder_path_out):
    joint_names = ['head', 'upper neck', 'right shoulder', 'right elbow', 'right wrist',
                   'left shoulder', 'left elbow', 'left wrist', 'right pelvis',
                   'right knee', 'right ankle', 'left pelvis', 'left knee', 'left ankle',
                   'background' ]
    joint_pairs = [['head', 'upper neck', 'purple'],
                   ['upper neck', 'right shoulder', 'yellow'],
                   ['upper neck', 'left shoulder', 'yellow'],
                   ['right shoulder', 'right elbow', 'blue'],
                   ['right elbow', 'right wrist', 'green'],
                   ['left shoulder', 'left elbow', 'blue'],
                   ['left elbow', 'left wrist', 'green'],
                   ['right shoulder', 'right pelvis', 'yellow'],
                   ['left shoulder', 'left pelvis', 'yellow'],
                   ['right pelvis', 'right knee', 'red'],
                   ['right knee', 'right ankle', 'skyblue'],
                   ['left pelvis', 'left knee', 'red'],
                   ['left knee', 'left ankle', 'skyblue']]

    # setup paths
    deployFile = deploy_proto_path
    caffemodel = caffe_model_prefix + '.caffemodel'
    norm_size = img_size
    print('deployFile = %s\n', deployFile)
    print('caffemodel = %s\n', caffemodel)

    # load network
    if flag_GPU is True:
        caffe.set_mode_gpu()
        caffe.set_device(flag_GPU_id)
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(deployFile, caffemodel, caffe.TEST)

    print('testing scales: ', scales)
    print('output_image_folder_path: ', output_image_folder_path)

    # Do the batch test
    subfolder_names = get_immediate_subfolder_names(folder_path_in)
    for subfolder_name in subfolder_names:
        subfolder_path_in = os.path.join(folder_path_in, subfolder_name)
        img_names = get_immediate_childfile_names(subfolder_path_in)

        for ct, img_name in enumerate(img_names):
            if flag_selective is True and img_name not in IMG_NAMES_TO_SHOW: continue
            print("processing: ", img_name, ct, len(img_names))

            # load image
            img_path_in = os.path.join(subfolder_path_in, img_name)
            if is_image(img_path_in):  img_raw = cv2.imread(img_path_in)

            # find heatmaps
            # OPTIONAL: flip and zoom
            if flag_flip is True and flag_zoom is True:
                heatmaps_from_multi_res = process_img_scales_and_flips(net, img_raw, norm_size, scales, heatmap_layer_name)
            # OPTIONAL: flip
            elif flag_flip is True:
                heatmaps_from_multi_res = [process_img_flip(net, img_raw, norm_size),
                                           img_to_heatmaps(net, img_raw, norm_size, heatmap_layer_name)]
            # OPTIONAL: zoom
            elif flag_zoom is True:
                heatmaps_from_multi_res = process_img_scales(net, img_raw, norm_size, scales)
            else:
                heatmaps = img_to_heatmaps(net, img_raw, norm_size, heatmap_layer_name)

            # Average heatmaps
            if flag_flip is True or flag_zoom is True:
                heatmaps = average_multiple_heatmap_sets(heatmaps_from_multi_res)

            # Find joints from heatmaps
            if flag_nms is True:
                joints = find_joints_in_heatmaps_nms(heatmaps)
            else:
                joints = find_joints_in_heatmaps(heatmaps)

            # Draw joints on image
            img_demo = cv2.resize(img_raw, (norm_size, norm_size))
            img_demo = demo_poses_in_img(img_demo, joints, joint_pairs, joint_names)

            # Save images to a path
            if flag_save_images is True:
                create_folder(os.path.join(folder_path_out, subfolder_name))
                img_name_out = os.path.splitext(img_name)[0] + '.png'
                img_path_out = os.path.join(folder_path_out, subfolder_name, img_name_out)
                cv2.imwrite(img_path_out, img_demo)

            # Demo Heatmaps
            if flag_demo_heatmaps is True:
                demo_heatmaps(heatmaps, joint_names)

            # Save joint predictions for MPII dataset evaluation
            if flag_save_evaluation is True:
                save_folder = os.path.join(output_image_folder_path, 'quantitative')
                create_folder(save_folder)
                if test_dataset == 'test_MPII' or test_dataset == 'validate_MPII':
                    rect_id = find_rect_id(img_name)
                    save_pose_preditions(joints, save_folder, img_name, rect_id)
                elif test_dataset == 'test_LSP':
                    save_pose_preditions(joints, save_folder, img_name)


if __name__ == "__main__":
    ''' 0. Fast setting '''
    flag_fast_mode = False

    ''' 1. Choose generic settings '''
    # Choose GPU/CPU mode
    flag_GPU = True
    flag_GPU_id = 0
    # Choose how to test an image
    flag_nms = True
    flag_flip = True
    flag_zoom = True
    scales = [1, 0.75] #zoom scales if zoom is true
    # Choose what to display, if any
    flag_demo_poses = True
    flag_demo_heatmaps = False
    flag_selective = False
    IMG_NAMES_TO_SHOW= ['im1097.jpg']
    # Choose what and how to draw joints and connections
    flag_only_draw_sure = False
    flag_color_sticks = True
    # Choose what to save
    flag_save_images = False
    flag_save_evaluation = True

    if flag_fast_mode is True:
        flag_nms = False
        flag_flip = False
        flag_zoom = False
        flag_save_images = False
        flag_demo_heatmaps = False
        flag_demo_poses = False

    ''' 2. Choose what dataset to test on '''
    test_LSP = True
    validate_MPII = False
    test_MPII = False
    test_custom = False
    test_datasets = ['test_LSP', 'validate_MPII', 'test_MPII', 'test_custom']

    ''' 3. Choose what network to use '''
    test_net_index = 0
    test_nets = ['guided-fractal', 'stacked-hourglass']
    test_net = test_nets[test_net_index]

    if test_net == 'guided-fractal':
        deploy_proto_path =  '../training/GNet/prototxt/GNet_deploy.prototxt'
        caffe_model_prefix = '../models/guided-fractal/GNet'
        img_size = 256
        heatmap_size = 64
        heatmap_layer_name = 'heatmap2'
    elif test_net == 'stacked-hourglass':
        deploy_proto_path =  '../training/GNet/prototxt/hourglass_net_deploy.prototxt'
        caffe_model_prefix = '../models/stacked-hourglass/stacked-hourglass'
        img_size = 256
        heatmap_size = 64
        heatmap_layer_name = 'heatmap2'

    for test_dataset in test_datasets:
        if test_dataset == 'test_LSP' and test_LSP is True:
            ''' LSP test set '''
            test_image_folder_path = "dataset_lsp/images_cropped/"
            output_image_folder_path = 'dataset_lsp/results_cropped/'
            create_folder(output_image_folder_path)
            batch_test_images(test_image_folder_path, output_image_folder_path)

        elif test_dataset == 'validate_MPII' and validate_MPII is True:
            ''' MPII validation set '''
            test_image_folder_path = "dataset_mpii/images_validate_cropped/"
            output_image_folder_path = 'dataset_mpii/results_validate_cropped/'
            create_folder(output_image_folder_path)
            batch_test_images(test_image_folder_path, output_image_folder_path)

        elif test_dataset == 'test_MPII' and test_MPII is True:
            ''' MPII test set '''
            test_image_folder_path = "dataset_mpii/images_test_cropped/"
            output_image_folder_path = 'dataset_mpii/results_test_cropped/'
            create_folder(output_image_folder_path)
            batch_test_images(test_image_folder_path, output_image_folder_path)

        elif test_dataset == 'test_custom' and test_custom is True:
            ''' Custom (for debug) '''
            test_image_folder_path = "dataset_custom/analysis_images/"
            output_image_folder_path = "dataset_custom/analysis_results/"
            create_folder(output_image_folder_path)
            batch_test_images(test_image_folder_path, output_image_folder_path)
