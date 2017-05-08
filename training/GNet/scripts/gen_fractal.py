'''
Automatically Generate <fractal networks>.
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Feb, 2017
'''
import sys
import os, math
from gen_hourglass import *

def main():
    hourglass_net_str = fractal_net_full()
    with open("../prototxt/fractal_train.prototxt", "w") as f:
        f.write(hourglass_net_str)


''' Subnetwork-level '''
def fractal_net_full(hourglass_level = 4, data_prototxt_path = 'data.prototxt'):
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

    return hourglass_net_str


def fractal_net(hourglass_level = 4, data_prototxt_path = 'data.prototxt'):
    hourglass_net_str = ''

    ''' Load data-specific part, which is specified and hand-written by human '''
    hourglass_net_data_str = load_str_from_prototxt(data_prototxt_path)
    hourglass_net_str += hourglass_net_data_str

    conv_layer_str = conv_layer('conv_1', 'image', 'conv_1', 64, 7, 2)
    hourglass_net_str += conv_layer_str

    batchnorm_layer_str = batch_norm_layer('bn_1', 'scale_1', 'conv_1')
    hourglass_net_str += batchnorm_layer_str

    residual_block_str = residual_block('residual_1', 'scale_1', 128)
    hourglass_net_str += residual_block_str

    pooling_layer_str = pooling_layer('pooling', 'residual_1')
    hourglass_net_str += pooling_layer_str

    residual_block_str = residual_block('residual_2', 'pooling', 128)
    hourglass_net_str += residual_block_str

    residual_block_str = residual_block('residual_3', 'residual_2', 128)
    hourglass_net_str += residual_block_str

    residual_block_str = residual_block('residual_4', 'residual_3', 256)
    hourglass_net_str += residual_block_str

    hourglass_module_str = fractal_module('hourglass1', 'residual_4', 512, hourglass_level)
    hourglass_net_str += hourglass_module_str

    linear_layer_str = conv_layer('fc_1', 'hourglass1', 'fc_1', 512, 1, 1)
    hourglass_net_str += linear_layer_str

    linear_layer_str = conv_layer('fc_2', 'fc_1', 'fc_2', 256, 1, 1)
    hourglass_net_str += linear_layer_str

    conv_layer_str = conv_layer('conv_2', 'fc_2', 'heatmap', 15, 1, 1)
    hourglass_net_str += conv_layer_str

    slice_layer_str = slice_layer('heatmap_slice', 'heatmap', 'heatmap_joints', 'heatmap_background')
    hourglass_net_str += slice_layer_str

    loss_layer_str = loss_layer('loss_joints', 'heatmap_joints', 'label_lower_joints')
    hourglass_net_str += loss_layer_str

    loss_layer_str = loss_layer('loss_background', 'heatmap_background', 'label_lower_background', 0.05)
    hourglass_net_str += loss_layer_str

    return hourglass_net_str


''' Module-level '''
def fractal_module(hourglass_module_name, bottom_layer_name, num_output_channels, hourglass_level):
    ''' There are 2 branches in a fractal module (same as hourglass):
        (1) Branch_1: Down-sampling and then Up-sampling, in order to extract multi-scale features
            Between Down-sampling and Up-sampling, there are 4 residual blocks and 1 lower-level hourglass_module
            For residual blocks, channels are: [M, 256], [256, 256], [256, 256], [N, N]
            For lower-level hourglass_module, channel is: [256, N], it is between the 3rd and 4th residual blocks.
        (2) Branch_2: Keey Original-Scale information
            3 inception-residual blocks in this branch
            channels are: [M, 256], [256, 256], [256, N]
    '''
    hourglass_module_str = ''
    N = num_output_channels
    num_channels_out_list = [[256, 256, 256, N, N], [256, 256, N]]

    ''' (1) branch 1 '''
    ''' 1.1 pooling '''
    pooling_layer_name = hourglass_module_name + '_pooling'
    pooling_layer_str = pooling_layer(pooling_layer_name, bottom_layer_name)
    hourglass_module_str += pooling_layer_str

    ''' 1.2 residual blocks '''
    for res_time in range(1, 4):
        residual_block_name = hourglass_module_name + '_branch1_res' + str(res_time)
        residual_block_name_last = hourglass_module_name + '_branch1_res' + str(res_time -1)

        bottom_name = pooling_layer_name if res_time == 1 else residual_block_name_last
        num_channels_out = num_channels_out_list[0][res_time -1]
        residual_block_str = inception_resnet_revised_block(residual_block_name,
                                            bottom_name,
                                            num_channels_out)
        hourglass_module_str += residual_block_str

    ''' 1.3 lower-level hourglass '''
    hourglass_lower_level_name = hourglass_module_name + '_branch1_level' + str(hourglass_level)
    bottom_name = residual_block_name
    num_channels_out = num_channels_out_list[0][3]
    if hourglass_level == 1:
        hourglass_lower_level_str = inception_resnet_revised_block(hourglass_lower_level_name,
                                                   bottom_name,
                                                   num_channels_out)
    else:
        hourglass_lower_level_str = hourglass_module(hourglass_lower_level_name,
                                                         bottom_name,
                                                         num_channels_out,
                                                         hourglass_level - 1)
        #hourglass_lower_level_str = fractal_module(hourglass_lower_level_name,
                                                        #  bottom_name,
                                                        #  num_channels_out,
                                                        #  hourglass_level - 1)
    hourglass_module_str += hourglass_lower_level_str


    ''' 1.4 residual block '''
    residual_block_name = hourglass_module_name + '_branch1_res5'
    bottom_name = hourglass_lower_level_name
    num_channels_out = num_channels_out_list[0][4]
    residual_block_str = inception_resnet_revised_block(residual_block_name,
                                        bottom_name,
                                        num_channels_out)
    hourglass_module_str += residual_block_str

    ''' 1.5 upsampling '''
    upsampling_layer_name = hourglass_module_name + '_upsampling'
    bottom_name = residual_block_name
    upsampling_factor = 2
    upsampling_layer_str = upsampling_layer(upsampling_layer_name,
                                            bottom_name,
                                            N, upsampling_factor)
    hourglass_module_str += upsampling_layer_str
    hourglass_name_br1 = upsampling_layer_name

    ''' (2) branch 2 '''
    for res_time in range(1, 4):
        residual_block_name = hourglass_module_name + '_branch2_res' + str(res_time)
        residual_block_name_last = hourglass_module_name + '_branch2_res' + str(res_time -1)

        bottom_name = bottom_layer_name if res_time == 1 else residual_block_name_last
        num_channels_out = num_channels_out_list[1][res_time -1]
        residual_block_str = inception_resnet_revised_block(residual_block_name,
                                            bottom_name,
                                            num_channels_out)
        hourglass_module_str += residual_block_str
    hourglass_name_br2 = residual_block_name

    ''' (3) Pixel-wise addition of two branches '''
    eltwise_layer_name = hourglass_module_name
    eltwise_layer_str = eltwise_layer(eltwise_layer_name,
                                      hourglass_name_br1,
                                      hourglass_name_br2 )

    hourglass_module_str += eltwise_layer_str

    return hourglass_module_str


def inception_resnet_revised_block(residual_block_name, bottom_layer_name, num_output_channels):
    ''' There are 2 branches in a residual block:
        (1) Branch_1: Extract higher-level information
            2 child branches:
                             a) 3 convolutional layers in this branch,
                                the channels are : [M, N/2], [N/2, N/2], [N/2, N/2]
                                the kernel sizes are: [1], [3], [3]
                             b) 2 convolutional layers in this branch,
                                the channels are : [M, N/2], [N/2, N/2]
                                the kernel sizes are: [1], [3]
            the 2 child branches are concatenated channel-wise, after which going
            through a convolutional layer with linear activation functions
            the channel is: [N]
            the kernel size is: [1]
        (2) Branch_2: Keep Original Low-level information
            1 convolutional layer in this branch,
            the channel is: [M, N]
            the kernel size is: [1]
        For branch_1, each conv layer is preceded with batch norm and relu layers.
        The two branches will be added element-wise as the output of the block.
    '''
    residual_block_str = ''
    N = num_output_channels
    num_channels_out_list = [[N/2, N/2, N/2], [N/2, N/2]]
    kernel_size_list = [[1, 3, 3], [1, 3]]
    child_branch_ranges = [[1, 2, 3], [1, 2]]

    ''' Branch 1: child branch 1 '''
    for child_branch in range(1, 3):
        for conv_time in child_branch_ranges[child_branch - 1]:
            bn_name = '_bn' + str(conv_time)
            scale_name = '_scale' + str(conv_time)
            relu_name = '_relu' + str(conv_time)
            conv_name = '_conv' + str(conv_time)
            conv_name_last = '_conv' + str(conv_time - 1)

            batch_norm_layer_name = residual_block_name + '_branch1' + '_sub' + str(child_branch) + bn_name
            scale_layer_name = residual_block_name + '_branch1'  + '_sub' + str(child_branch) + scale_name
            relu_layer_name = residual_block_name + '_branch1' + '_sub' + str(child_branch) + relu_name
            conv_layer_name_br1 = residual_block_name + '_branch1' + '_sub' + str(child_branch) + conv_name
            conv_layer_name_br1_last = residual_block_name + '_branch1' + '_sub' + str(child_branch) + conv_name_last

            ''' batch norm '''
            if conv_time == 1:
                bottom_name = bottom_layer_name
            else:
                bottom_name = conv_layer_name_br1_last
            batch_norm_layer_str = batch_norm_layer(batch_norm_layer_name,
                                                        scale_layer_name,
                                                        bottom_name)
            residual_block_str += batch_norm_layer_str

            ''' relu '''
            bottom_name = scale_layer_name
            relu_layer_str = relu_layer(relu_layer_name,
                                        bottom_name)
            residual_block_str += relu_layer_str

            ''' conv '''
            bottom_name = relu_layer_name
            top_name = conv_layer_name_br1
            num_channels_out = num_channels_out_list[child_branch - 1][conv_time -1]
            kernel_size = kernel_size_list[child_branch - 1][conv_time - 1]
            conv_layer_str = conv_layer(conv_layer_name_br1,
                                        bottom_name,
                                        top_name,
                                        num_channels_out,
                                        kernel_size)
            residual_block_str += conv_layer_str

            if child_branch == 1 and conv_time == 3:
                conv_layer_name_br1_sub1_top = top_name
            elif child_branch == 2 and conv_time == 2:
                conv_layer_name_br1_sub2_top = top_name
    ''' Channel-wise Concatenation of Branch 1's two sub-branches '''
    concat_layer_name = residual_block_name + '_branch1' + '_concat'
    concat_layer_str = concat_layer(concat_layer_name,
                                    conv_layer_name_br1_sub1_top,
                                    conv_layer_name_br1_sub2_top)
    residual_block_str += concat_layer_str
    ''' conv '''
    conv_layer_name_br1 = residual_block_name + '_branch1_conv_linear'
    bottom_name = concat_layer_name
    top_name = conv_layer_name_br1
    num_channels_out = N
    kernel_size = 1
    conv_layer_str = conv_layer(conv_layer_name_br1,
                                bottom_name,
                                top_name,
                                num_channels_out,
                                kernel_size)
    residual_block_str += conv_layer_str
    ''' batch norm ''' # Newly added
    bottom_name = conv_layer_name_br1
    batch_norm_layer_name = residual_block_name + '_branch1' + '_linear_batch'
    scale_layer_name = residual_block_name + '_branch1'  + '_linear_scale'
    batch_norm_layer_str = batch_norm_layer(batch_norm_layer_name,
                                           scale_layer_name,
                                           bottom_name)
    residual_block_str += batch_norm_layer_str
    conv_layer_name_br1 = scale_layer_name


    ''' Branch 2 '''
    conv_layer_name_br2 = residual_block_name + '_branch2_conv'
    bottom_name = bottom_layer_name
    top_name = conv_layer_name_br2
    num_channels_out = N
    kernel_size = 1
    conv_layer_str = conv_layer(conv_layer_name_br2,
                                bottom_name,
                                top_name,
                                num_channels_out,
                                kernel_size)
    residual_block_str += conv_layer_str

    ''' element-wise addition '''
    eltwise_layer_name = residual_block_name
    eltwise_layer_str = eltwise_layer(eltwise_layer_name,
                                      conv_layer_name_br1,
                                      conv_layer_name_br2 )

    residual_block_str += eltwise_layer_str

    return residual_block_str


if __name__ == "__main__":
    main()
