% Author: Guanghan Ning, Dec 2016
clear all;
clc;
close all;
addpath('../utils_eval_mpii/');
addpath('../eval_MPII/');
mat_folder_path = '../dataset_mpii/results_all_validate_cropped/quantitative/';

% Load annotations: validation
load('../eval_MPII/mpii_Tompson_validation', 'annolist');
load('../eval_MPII/mpii_Tompson_validation', 'RELEASE_img_index');
load('../eval_MPII/mpii_Tompson_validation', 'RELEASE_person_index');
annolist_validate = annolist;
num_validate_imgs = size(annolist_validate, 2);

% Load annotations: train + validation
load('../eval_MPII/mpii_human_pose_v1_u12_1','RELEASE');
annolist_all = RELEASE.annolist;

% Load scales extracted from image height
scale_in_cpp = load('../scale_in_cpp_MPII_validate.mat');
scale_in_cpp = scale_in_cpp.output_scale;

% Feed our predictions to the annotations mat file
pred_validation = zeros(2, 14, num_validate_imgs);

%% Start from here if everything already read from mat file.
for img_id = 1:num_validate_imgs
    
    img_index = RELEASE_img_index(img_id);
    img_name = annolist_all(img_index).image.name;

        % Get rect id. Note the difference between rect_ct and rect_id
        rect_id = RELEASE_person_index(img_id);

        % Find scale and mid-points in order to convert prediction
        x_mid = annolist_validate(img_id).annorect.objpos.x;
        y_mid = annolist_validate(img_id).annorect.objpos.y;
        mid_point = [x_mid, y_mid];  %scale = annolist_test(img_id).annorect.scale;

        % Load our algorithm predictions for this rect
        [folder_path, base_name, extension] = fileparts(img_name);
        [x_preds_raw, y_preds_raw] = load_pred_result(mat_folder_path, base_name, rect_id);

        for joint_id = 1:14
            % Convert predictions for un-cropped image scale for evaluation
            %id = convert_joint_order_to_Tompson_order(joint_id);
            id = convert_joint_order_to_gt_order(joint_id);

            pred_point_raw = [x_preds_raw(joint_id), y_preds_raw(joint_id)];
            [x_pred, y_pred]= convert_trans_and_scale_validation(pred_point_raw, scale_in_cpp(img_id), mid_point);
            
            % Or we can just make [pred] the format of [2 x 14 x N]  
            pred_validation(1, id, img_id) = x_pred;
            pred_validation(2, id, img_id) = y_pred;
        end
end

save('../eval_MPII/pred/ning17iccv/pred_keypoints_mpii_validate.mat','pred_validation');

%% Call Official evaluatePCKh() function to process the evaluation
evaluatePCKh_validation(1, false);
