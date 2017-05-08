% Author: Guanghan Ning, Feb 2017
clear all;
clc;
close all;
addpath('../utils_eval_lsp/');
addpath('../eval_LSP/');
mat_folder_path = '../dataset_lsp/results_cropped/quantitative/';
img_folder_path = '../../dataset/LEEDS/lsp_dataset/images/';

scale_in_cpp = load('../scale_in_cpp_LSP.mat');
scale_in_cpp = scale_in_cpp.output_scale;

pred = zeros(2, 14, 1000);
num_test_imgs = 1000;

for img_id = 1:num_test_imgs

    base_name = strcat('im', num2str(img_id + 1000));
    img_path = strcat(img_folder_path, '/', base_name, '.jpg');
    img_info = imfinfo(img_path);
    org_img_size = [img_info.Width, img_info.Height];

    [x_preds_raw, y_preds_raw] = load_pred_result(mat_folder_path, base_name);

    for joint_id = 1:14 %joint_id in our joint order

        % Convert joint to lsp-defined order for evaluation
        id = convert_joint_order(joint_id);  % id in LSP order

        % Convert predictions from 256 to image-size for evaluation
        pred_point_raw = [x_preds_raw(joint_id), y_preds_raw(joint_id)];
        [x_pred, y_pred]= convert_scale_cropped(pred_point_raw, org_img_size, scale_in_cpp(img_id));

        pred(1, id, img_id) = x_pred;
        pred(2, id, img_id) = y_pred;
    end
end
save('../eval_LSP/pred/ning17iccv/pred_keypoints_lsp.mat','pred');

%% Call Official evaluatePCKh() function to process the evaluation
evaluatePCK(33,'PC', true); % 33 is my model index, using person-centric (PC)
