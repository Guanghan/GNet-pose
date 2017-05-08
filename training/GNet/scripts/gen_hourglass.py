'''
Automatically Generate <stacked-hourglass>.
    Author: Guanghan Ning
    E-mail: gnxr9@mail.missouri.edu
    Dec, 2016
'''
import sys
import os, math

def main():
    hourglass_net_str = hourglass_net_full()
    with open("../prototxt/hourglass_train.prototxt", "w") as f:
        f.write(hourglass_net_str)


''' Network-level'''
def assemble_parts(part_str_list):
    net_str = ''
    for part_str in part_str_list:
        net_str += part_str
    return net_str


''' Subnetwork-level '''
def hourglass_net_full(hourglass_level = 4, data_prototxt_path = 'data.prototxt'):
    hourglass_net_str = hourglass_net(hourglass_level, data_prototxt_path)

    ''' Additional net structure '''
    conv_layer_str = conv_layer('conv_3', 'heatmap', 'conv_3', 384, 1, 1)
    hourglass_net_str += conv_layer_str

    concat_layer_str = concat_layer('concat_1', 'pooling', 'fc_2')
    hourglass_net_str += concat_layer_str

    conv_layer_str = conv_layer('conv_4', 'concat_1', 'conv_4', 384, 1, 1)
    hourglass_net_str += conv_layer_str

    eltwise_layer_str = eltwise_layer('eltwise', 'conv_3', 'conv_4')
    hourglass_net_str += eltwise_layer_str

    hourglass_module_str = hourglass_module('hourglass2', 'eltwise', 512, hourglass_level)
    hourglass_net_str += hourglass_module_str

    #linear_layer_str = linear_layer('fc_3', 'hourglass2', 512)
    #hourglass_net_str += linear_layer_str

    #linear_layer_str = linear_layer('fc_4', 'fc_3', 256)
    #hourglass_net_str += linear_layer_str

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


def hourglass_net(hourglass_level = 4, data_prototxt_path = 'data.prototxt'):
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

    hourglass_module_str = hourglass_module('hourglass1', 'residual_4', 512, hourglass_level)
    hourglass_net_str += hourglass_module_str

    #linear_layer_str = linear_layer('fc_1', 'hourglass1', 512)
    #hourglass_net_str += linear_layer_str

    #linear_layer_str = linear_layer('fc_2', 'fc_1', 256)
    #hourglass_net_str += linear_layer_str

    # Here: fc_1 and fc_2 are not fully connected, but conv layers with linear activation functions

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


def load_str_from_prototxt(prototxt_path):
    if os.path.isfile(prototxt_path) is not True:
        print("prototxt file does not exist!\n")
        return ''
    with open(prototxt_path, 'r') as f:
        data_str = f.read()
    return data_str


''' Module-level '''
def hourglass_module(hourglass_module_name, bottom_layer_name, num_output_channels, hourglass_level):
    ''' There are 2 branches in an hourglass module:
        (1) Branch_1: Down-sampling and then Up-sampling, in order to extract multi-scale features
            Between Down-sampling and Up-sampling, there are 4 residual blocks and 1 lower-level hourglass_module
            For residual blocks, channels are: [M, 256], [256, 256], [256, 256], [N, N]
            For lower-level hourglass_module, channel is: [256, N], it is between the 3rd and 4th residual blocks.
        (2) Branch_2: Keey Original-Scale information
            3 residual blocks in this branch
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
        residual_block_str = residual_block(residual_block_name,
                                            bottom_name,
                                            num_channels_out)
        hourglass_module_str += residual_block_str

    ''' 1.3 lower-level hourglass '''
    hourglass_lower_level_name = hourglass_module_name + '_branch1_level' + str(hourglass_level)
    bottom_name = residual_block_name
    num_channels_out = num_channels_out_list[0][3]
    if hourglass_level == 1:
        hourglass_lower_level_str = residual_block(hourglass_lower_level_name,
                                                   bottom_name,
                                                   num_channels_out)
    else:
        hourglass_lower_level_str = hourglass_module(hourglass_lower_level_name,
                                                         bottom_name,
                                                         num_channels_out,
                                                         hourglass_level - 1)
    hourglass_module_str += hourglass_lower_level_str


    ''' 1.4 residual block '''
    residual_block_name = hourglass_module_name + '_branch1_res5'
    bottom_name = hourglass_lower_level_name
    num_channels_out = num_channels_out_list[0][4]
    residual_block_str = residual_block(residual_block_name,
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
        residual_block_str = residual_block(residual_block_name,
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


''' Block-level '''
def residual_block(residual_block_name, bottom_layer_name, num_output_channels):
    ''' There are 2 branches in a residual block:
        (1) Branch_1: Extract higher-level information
            3 convolutional layers in this branch,
            the channels are : [M, N/2], [N/2, N/2], [N/2, N]
            the kernel sizes are: [1], [3], [1]
        (2) Branch_2: Keep Original Low-level information
            1 convolutional layer in this branch,
            the channel is: [M, N]
            the kernel size is: [1]
        For branch_1, each conv layer is preceded with batch norm and relu layers.
        The two branches will be added element-wise as the output of the block.
    '''
    residual_block_str = ''
    N = num_output_channels
    num_channels_out_list = [N/2, N/2, N]
    kernel_size_list = [1, 3, 1]

    ''' Branch 1 '''
    for conv_time in range(1, 4):
        bn_name = '_bn' + str(conv_time)
        scale_name = '_scale' + str(conv_time)
        relu_name = '_relu' + str(conv_time)
        conv_name = '_conv' + str(conv_time)
        conv_name_last = '_conv' + str(conv_time - 1)

        batch_norm_layer_name = residual_block_name + '_branch1' + bn_name
        relu_layer_name = residual_block_name + '_branch1' + relu_name
        conv_layer_name_br1 = residual_block_name + '_branch1' + conv_name
        conv_layer_name_br1_last = residual_block_name + '_branch1' + conv_name_last

        ''' batch norm '''
        scale_layer_name = residual_block_name + '_branch1' + scale_name
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
        num_channels_out = num_channels_out_list[conv_time -1]
        kernel_size = kernel_size_list[conv_time - 1]
        conv_layer_str = conv_layer(conv_layer_name_br1,
                                    bottom_name,
                                    top_name,
                                    num_channels_out,
                                    kernel_size)
        residual_block_str += conv_layer_str

    ''' Branch 2 '''
    conv_layer_name_br2 = residual_block_name + '_branch2_conv'
    bottom_name = bottom_layer_name
    top_name = conv_layer_name_br2
    num_channels_out = num_channels_out_list[2]
    kernel_size = kernel_size_list[2]
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


''' Layer-level '''
def concat_layer(layer_name, bottom_name_1, bottom_name_2):
    layer_type = 'Concat'
    layer_bottom_name = bottom_name_1
    layer_top_name = layer_name
    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)

    layer_body_str += ' bottom: {}\n'.format('"' + bottom_name_2 + '"')
    layer_body_str += ' concat_param { \n' +  ' axis: 1\n } \n'
    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def slice_layer(layer_name, bottom_name, top_name_1, top_name_2, slice_point = 14):
    layer_type = 'Slice'
    layer_bottom_name = bottom_name
    layer_top_name = top_name_1
    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)

    layer_body_str += ' top: {}\n'.format('"' + top_name_2 + '"')
    layer_body_str += (' slice_param { \n' + ' slice_point: {}'.format(str(slice_point) + '\n axis: 1\n }\n'))
    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def loss_layer(layer_name, bottom_name_1, bottom_name_2, loss_weight = 1, loss_type = 'EuclideanLoss'):
    layer_type = loss_type
    layer_bottom_name = bottom_name_1
    layer_top_name = layer_name
    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)

    layer_body_str += ' bottom: {}\n'.format('"' + bottom_name_2 + '"')
    layer_body_str += ' loss_weight: {}\n'.format(str(loss_weight))
    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def linear_layer(layer_name, bottom_name, num_output_channels):
    layer_type = 'InnerProduct'
    layer_bottom_name = bottom_name
    layer_top_name = layer_name
    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)

    inner_product_param_str = inner_product_param(num_output_channels)
    layer_lr_param_str = lr_param(weight_lr = [1, 1], bias_lr = [2, 0])

    layer_body_str += inner_product_param_str
    layer_body_str += layer_lr_param_str
    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def eltwise_layer(layer_name, bottom_name_1, bottom_name_2):
    layer_type = 'Eltwise'
    layer_bottom_name_1 = bottom_name_1
    layer_top_name = layer_name
    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name_1, layer_top_name)

    layer_body_str += ' bottom: {} \n '.format('"' + bottom_name_2 + '"')
    layer_body_str += 'eltwise_param { \n operation: SUM \n } \n '

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def batch_norm_layer(bn_name, scale_name, bottom_name, param_name = None):
    ''' deal with batch norm '''
    layer_type = 'BatchNorm'
    layer_name = bn_name
    layer_bottom_name = bottom_name
    layer_top_name = bn_name
    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)

    if param_name is None:
        param_str = 'param { \n lr_mult: 0 \n } \n'
        layer_body_str += ' {} {} {}'.format(param_str, param_str, param_str)
    else:
        for num in range(3):
            param_name_this = param_name + '_' + str(num + 1)
            param_str = 'param { \n' + 'name: ' + param_name_this + '\n lr_mult: 0 \n } \n'
            layer_body_str += ' {} '.format(param_str)

    layer_str = add_str_head_and_end(layer_body_str)

    ''' deal with scale '''
    layer2_type = 'Scale'
    layer2_name = scale_name
    layer2_bottom_name = bn_name
    layer2_top_name = scale_name
    layer2_body_str = init_str_body(layer2_name, layer2_type, layer2_bottom_name, layer2_top_name)

    scale_param_str = ' scale_param { \n bias_term: true \n } \n '
    layer2_body_str += scale_param_str

    layer2_str = add_str_head_and_end(layer2_body_str)

    return layer_str + layer2_str


def upsampling_layer(layer_name, bottom_name, num_channels_out, upsampling_factor):
    layer_type = 'Deconvolution'
    layer_body_str = init_str_body(layer_name, layer_type, bottom_name, layer_name)

    layer_lr_param_str = 'param {' + '\n \t lr_mult: {}\n \t decay_mult: {}\n '.format(0, 0) + '}\n'
    deconvolution_param_str = deconvolution_param(num_channels_out, upsampling_factor)
    layer_body_str += ' {} {} '.format(deconvolution_param_str, layer_lr_param_str)

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def deconv_layer(layer_name, bottom_name, top_name, num_channels_out, kernel_size):
    layer_type = 'Deconvolution'
    layer_body_str = init_str_body(layer_name, layer_type, bottom_name, top_name)

    layer_lr_param_str = lr_param()
    convolution_param_str = convolution_param(num_channels_out, kernel_size)
    layer_body_str += ' {} {} '.format(layer_lr_param_str, convolution_param_str)

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def conv_layer(layer_name, bottom_name, top_name, num_channels_out, kernel_size, stride_size = 1):
    layer_type = 'Convolution'
    layer_body_str = init_str_body(layer_name, layer_type, bottom_name, top_name)

    layer_lr_param_str = lr_param()
    convolution_param_str = convolution_param(num_channels_out, kernel_size, stride_size)
    layer_body_str += ' {} {} '.format(layer_lr_param_str, convolution_param_str)

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def relu_layer(relu_name, bottom_name, flag_leaky = False):
    layer_type = 'ReLU'
    layer_name = relu_name
    layer_bottom_name = bottom_name
    layer_top_name = relu_name

    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)
    if flag_leaky is True:
        pooling_param = ' relu_param { \n negative_slope: 0.1 \n } \n '
        layer_body_str += pooling_param

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def tanh_layer(tanh_name, bottom_name):
    layer_type = 'TanH'
    layer_name = tanh_name
    layer_bottom_name = bottom_name
    layer_top_name = tanh_name

    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)
    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def sigmoid_layer(sigmoid_name, bottom_name):
    layer_type = 'Sigmoid'
    layer_name = sigmoid_name
    layer_bottom_name = bottom_name
    layer_top_name = tanh_name

    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)
    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def pooling_layer(pool_name, bottom_name):
    layer_type = 'Pooling'
    layer_name = pool_name
    layer_bottom_name = bottom_name
    layer_top_name = pool_name

    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)
    pooling_param = ' pooling_param { \n pool: MAX \n kernel_size: 2 \n stride: 2 \n } \n '
    layer_body_str += pooling_param

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


def dropout_layer(dropout_layer_name, bottom_layer_name, dropout_rate = 0.5):
    layer_type = 'Dropout'
    layer_name = dropout_layer_name
    layer_bottom_name = bottom_layer_name
    layer_top_name = dropout_layer_name

    layer_body_str = init_str_body(layer_name, layer_type, layer_bottom_name, layer_top_name)
    dropout_param = ' dropout_param {' + '\n dropout_ratio: {}\n '.format(dropout_rate) + '}\n'
    layer_body_str += dropout_param

    layer_str = add_str_head_and_end(layer_body_str)
    return layer_str


''' Layer components '''
def lr_param(weight_lr = [1, 1], bias_lr = [2, 0]):
    layer_lr_param_str = ' param {' + '\n \t lr_mult: {}\n \t decay_mult: {}\n '.format(weight_lr[0], weight_lr[1]) + '}\n'
    layer_lr_param_str += ' param {' + '\n \t lr_mult: {}\n \t decay_mult: {}\n '.format(bias_lr[0], bias_lr[1]) + '}\n'

    return layer_lr_param_str


def weight_filler(filler_type = "gaussian", std_variance = 0.01):
    weight_filler_str = 'weight_filler { \n type: "' +  filler_type + '"\n std: ' + str(std_variance) + ' \n }\n'
    return weight_filler_str


def bias_filler(filler_type = "constant"):
    bias_filler_str = 'bias_filler {' + '\n type: {}\n '.format('"' + filler_type + '"') + '}'
    return bias_filler_str


def inner_product_param (num_channels_out,
                         weight_filler_str = weight_filler(),
                         bias_filler_str = bias_filler()):
    inner_product_param_header_str = ' inner_product_param { \n '
    inner_product_param_end_str = '\n }\n'
    inner_product_param_str_body = 'num_output: {}\n {} {}'.format(str(num_channels_out),
                                                                 weight_filler_str,
                                                                 bias_filler_str)
    inner_product_param_str = inner_product_param_header_str + inner_product_param_str_body + inner_product_param_end_str
    return inner_product_param_str


def convolution_param( num_channels_out,
                       kernel_size,
                       stride_size = 1,
                       weight_filler_str = weight_filler(),
                       bias_filler_str = bias_filler()):
    ''' automatic determine padding so that the map does not reduce in size '''
    padding_size = (kernel_size - 1)/2

    convolution_param_header_str = 'convolution_param { \n '
    convolution_param_end_str = '\n }\n'
    convolution_param_str_body = 'num_output: {}\n pad: {}\n kernel_size: {}\n stride:{}\n {} {}'.format(str(num_channels_out),
                                                                                             str(padding_size),
                                                                                             str(kernel_size),
                                                                                             str(stride_size),
                                                                                             weight_filler_str,
                                                                                             bias_filler_str)
    convolution_param_str = convolution_param_header_str + convolution_param_str_body + convolution_param_end_str
    return convolution_param_str


def deconvolution_param(num_channels_out,
                        upsampling_factor):
    padding_size = int(math.ceil((upsampling_factor - 1)/2.0))
    stride_size = upsampling_factor
    kernel_size = 2 * upsampling_factor - upsampling_factor % 2
    group_size = num_channels_out

    weight_filler_str = 'weight_filler: { type: "bilinear" }\n'
    bias_filler_str = 'bias_term: false'

    deconvolution_param_header_str = 'convolution_param { \n '
    deconvolution_param_end_str = '\n }\n'
    deconvolution_param_str_body = 'num_output: {}\n pad: {}\n kernel_size: {}\n stride: {}\n group: {}\n {} {}'.format(str(num_channels_out),
                                                                                             str(padding_size),
                                                                                             str(kernel_size),
                                                                                             str(stride_size),
                                                                                             str(group_size),
                                                                                             weight_filler_str,
                                                                                             bias_filler_str)
    deconvolution_param_str = deconvolution_param_header_str + deconvolution_param_str_body + deconvolution_param_end_str
    return deconvolution_param_str


''' Layer miscellaneous'''
def add_str_head_and_end(layer_body_str):
    layer_head_str = 'layer {\n'
    layer_end_str = '}\n'
    layer_str = layer_head_str + layer_body_str + layer_end_str
    return layer_str


def init_str_body(layer_name, layer_type, bottom_name, top_name):
    layer_body_str = ' name: {}\n type: {} \n bottom: {} \n top: {} \n'.format('"' + layer_name + '"',
                                                                                '"' + layer_type + '"',
                                                                                '"' + bottom_name + '"',
                                                                                '"' + top_name + '"')
    return layer_body_str


if __name__ == "__main__":
    main()
