'''
Automatically Generate <GNet: Guided Fractal Networks>.
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Feb, 2017
'''
import sys
import os, math
from gen_hourglass import *
from gen_fractal import *

def main():
    net_str = guided_fractal_net()
    with open("../prototxt/GNet_train.prototxt", "w") as f:
        f.write(net_str)


''' Subnetwork-level '''
def guided_fractal_net(hourglass_level = 4, data_prototxt_path = 'data_inject.prototxt'):
    hourglass_net_str = fractal_net(hourglass_level, data_prototxt_path)

    ''' Additional net structure '''
    conv_layer_str = conv_layer('conv_3', 'heatmap', 'conv_3', 384, 1, 1)
    hourglass_net_str += conv_layer_str

    concat_layer_str = concat_layer('concat_1', 'pooling', 'fc_2')
    hourglass_net_str += concat_layer_str

    conv_layer_str = conv_layer('conv_4', 'concat_1', 'conv_4', 384, 1, 1)
    hourglass_net_str += conv_layer_str

    eltwise_layer_str = eltwise_layer('eltwise', 'conv_3', 'conv_4')
    hourglass_net_str += eltwise_layer_str

    hourglass_module_str = fractal_module('hourglass2', 'eltwise', 512, hourglass_level)
    hourglass_net_str += hourglass_module_str

    linear_layer_str = conv_layer('fc_3', 'hourglass2', 'fc_3', 512, 1, 1)
    hourglass_net_str += linear_layer_str

    linear_layer_str = conv_layer('fc_4', 'fc_3', 'fc_4', 256, 1, 1)
    hourglass_net_str += linear_layer_str

    conv_layer_str = conv_layer('conv_5', 'fc_4', 'heatmap2', 15, 1, 1)
    hourglass_net_str += conv_layer_str

    slice_layer_str = slice_layer('heatmap2_slice', 'heatmap2', 'heatmap2_joints', 'heatmap2_background')
    hourglass_net_str += slice_layer_str

    loss_layer_str = loss_layer('loss2_joints', 'heatmap2_joints', 'label_lower_joints')
    hourglass_net_str += loss_layer_str

    loss_layer_str = loss_layer('loss2_background', 'heatmap2_background', 'label_lower_background', 0.05)
    hourglass_net_str += loss_layer_str

    ''' Injection-related net structure '''

    ''' 1. Geographical knowledge with background heatmaps '''
    conv_layer_str = conv_layer('conv_6', 'heatmap2_background', 'conv_6', 8, 1, 1)
    hourglass_net_str += conv_layer_str

    # inner product (i.e. fc) background heatmap to be same size as the injected feature, called the encoded feature
    fc_layer_str = linear_layer('encoded_background', 'conv_6', 224)
    hourglass_net_str += fc_layer_str

    # enforce loss between the encoded feature and the injected feature
    loss_layer_str = loss_layer('loss2_feature', 'encoded_background', 'injected_feature', 0.05)
    hourglass_net_str += loss_layer_str

    ''' 2. Edge features with joint heatmaps and original image'''
    pooling_layer_name = 'image_pooling'
    pooling_layer_str = pooling_layer('image_128', 'image')
    hourglass_net_str += pooling_layer_str

    pooling_layer_str = pooling_layer('image_64', 'image_128')
    hourglass_net_str += pooling_layer_str

    concat_layer_str = concat_layer('concat_edge', 'heatmap2_joints', 'image_64')
    hourglass_net_str += concat_layer_str

    conv_layer_str = conv_layer('conv_7', 'concat_edge', 'conv_7', 64, 1, 1)
    hourglass_net_str += conv_layer_str

    conv_layer_str = conv_layer('conv_8', 'conv_7', 'conv_8', 32, 1, 1)
    hourglass_net_str += conv_layer_str

    fc_layer_str = linear_layer('encoded_edge', 'conv_8', 64)
    hourglass_net_str += fc_layer_str

    # enforce loss between the encoded feature and the injected feature
    loss_layer_str = loss_layer('loss2_edge_feature', 'encoded_edge', 'injected_edge_feature', 0.005)
    hourglass_net_str += loss_layer_str

    return hourglass_net_str


if __name__ == "__main__":
    main()
