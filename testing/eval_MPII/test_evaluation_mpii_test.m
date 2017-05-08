% Author: Guanghan Ning, March 2017
clear all;
clc;
close all;
addpath('../utils_eval_mpii/');
addpath('../eval_MPII/');

target_dist = 1.45; %41/35;
mat_folder_path = '../dataset_mpii/results_test_cropped/quantitative/';

% Load all annotations: train + validation + test
load('../eval_MPII/mpii_human_pose_v1_u12_1','RELEASE');
annolist = RELEASE.annolist;
annolist_test = annolist(RELEASE.img_train == 0);

pred = annolist_test;
rectidxs = RELEASE.single_person(RELEASE.img_train == 0);
%% Start from here if everything already read from mat file.

num_of_single_persons = 0;
for imgidx = 1:6908
        % Get rect id. Note the difference between rect_ct and rect_id
        rect_id_set = rectidxs{imgidx, 1};
        num_of_rects = size(rect_id_set, 1);

        flag_empty = isempty(rect_id_set);
        if flag_empty == 1
            disp(rect_id_set);
            structSize = length(annolist_test(imgidx).annorect);
            num_of_rects = structSize;
            continue; %skip this empty one
        end

        for rect_num = 1:num_of_rects
            if flag_empty == 1
                rect_id = rect_num;
            else
                rect_id = rect_id_set(rect_num);
            end

            disp(sprintf('[img_id, rect_id, rect_num]: (%d, %d, %d)', imgidx, rect_id, num_of_rects));
            num_of_single_persons = num_of_single_persons +1;

            % Find scale and mid-points in order to convert prediction
            x_mid = annolist_test(imgidx).annorect(rect_id).objpos.x;
            y_mid = annolist_test(imgidx).annorect(rect_id).objpos.y;
            mid_point = [x_mid, y_mid];
            scale_provided = annolist_test(imgidx).annorect(rect_id).scale;
            scale_in_cpp = target_dist/scale_provided;
            img_name = annolist_test(imgidx).image.name;

            % Load our algorithm predictions for this rect
            [folder_path, base_name, extension] = fileparts(img_name);
            [x_preds_raw, y_preds_raw] = load_pred_result(mat_folder_path, base_name, rect_id);

            for joint_id = 1:14
                % Convert predictions for un-cropped image scale for evaluation
                id = get_official_id(joint_id);

                pred_point_raw = [x_preds_raw(joint_id), y_preds_raw(joint_id)];
                [x_pred, y_pred]= convert_trans_and_scale_validation(pred_point_raw, scale_in_cpp, mid_point);
                disp(sprintf('[x, y]: (%f, %f)', x_pred, y_pred));

                pred(imgidx).annorect(rect_id).annopoints.point(joint_id).x = x_pred;
                pred(imgidx).annorect(rect_id).annopoints.point(joint_id).y = y_pred;
                pred(imgidx).annorect(rect_id).annopoints.point(joint_id).id = id;
            end
         end
end

save('../eval_MPII/pred/ning17iccv/pred_keypoints_mpii.mat','pred');

%% Predictions need to be sent to [leonid at mpi-inf.mpg.de] and [eldar at mpi-inf.mpg.de] for evaluation
