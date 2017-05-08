% Author: Guanghan Ning, Dec 2016
clear all;
addpath('../utils_eval_mpii/');
addpath('../eval_MPII/');

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
scale_in_cpp = load('../scale_in_cpp_MPII_validate_bigger_1.4.mat');
scale_in_cpp = scale_in_cpp.output_scale;


%%
img_folder_path = '/home/ngh/dev/POSE/POSE-dev/dataset/MPI/images/';%'../../../dataset/MPI/images/';

for scale_id = 1
    for iter_id = 1176058
        mat_folder_path = strcat('../dataset_mpii/results_all_validate_cropped_bigger_1.4/', num2str(iter_id), '_', num2str(scale_id-1), '/quantitative/');
  
        for img_id = 1:num_validate_imgs

            img_index = RELEASE_img_index(img_id);
            img_name = annolist_all(img_index).image.name;
            img_path = strcat(img_folder_path, '/', img_name);
            img_org_full = imread(img_path);

            % Display predicted results on the original image
            figure;
            imshow(img_org_full);
            hold on;

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
                %id = convert_joint_order(joint_id);
                id = convert_joint_order_to_gt_order(joint_id);

                pred_point_raw = [x_preds_raw(joint_id), y_preds_raw(joint_id)];
                [x_pred, y_pred]= convert_trans_and_scale_validation(pred_point_raw, scale_in_cpp(img_id), mid_point);

                plot(x_pred, y_pred,'o');
            end
            hold off;
            pause;
        end
    end
end
