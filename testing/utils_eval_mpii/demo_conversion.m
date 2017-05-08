% Need to draw the estimated poses on the original poses, 
% and make sure that the prediction saved to mat file are correct
% otherwise the evaluation gives wrong results

% Author: Guanghan Ning, Dec 2016
clear all;
addpath('../utils_eval_mpii/');
addpath('../eval_MPII/');

% Load annotations
load('../eval_MPII/mpii_human_pose_v1_u12_1','RELEASE');
annolist = RELEASE.annolist;
annolist_test = annolist(RELEASE.img_train == 0);
rect_ids_list = RELEASE.single_person(RELEASE.img_train == 0);

% Feed our predictions to the annotations mat file
pred = annolist_test;
num_test_imgs = size(annolist_test, 2);

%%
mat_folder_path = '../dataset_mpii/results_all/quantitative/';
%iter_id = 1176058;
%scale_id = 1;
%mat_folder_path = strcat('../dataset_mpii/results_all_validate_cropped_bigger_1.4/', num2str(iter_id), '_', num2str(scale_id-1), '/quantitative/');
img_folder_path = '/home/ngh/dev/POSE/POSE-dev/dataset/MPI/images/';%'../../../dataset/MPI/images/';

for img_id = 24:100
    img_name = annolist_test(img_id).image.name;
    img_path = strcat(img_folder_path, '/', img_name);
    img_org_full = imread(img_path);
        
    % Display predicted results on the original image
    figure;
    imshow(img_org_full);
    hold on;
    
    rect_ids = rect_ids_list{img_id, 1};
    if isempty(rect_ids) == 1
        continue
    end

    for rect_ct = 1:size(rect_ids, 1)
        % Get rect id. Note the difference between rect_ct and rect_id
        rect_id = rect_ids(rect_ct);
        
        % Find scale and mid-points in order to convert prediction
        x_mid = annolist_test(img_id).annorect(rect_id).objpos.x;
        y_mid = annolist_test(img_id).annorect(rect_id).objpos.y;
        mid_point = [x_mid, y_mid]; 
        scale = annolist_test(img_id).annorect(rect_id).scale;
        
        % Load our algorithm predictions for this rect
        [folder_path, base_name, extension] = fileparts(img_name);
        
        mat_name = strcat(base_name, '_', num2str(rect_id), '.mat');
        mat_path = strcat(mat_folder_path, '/', mat_name);
        if exist(mat_path) == 0
            continue
        end
        
        [x_preds_raw, y_preds_raw] = load_pred_result(mat_folder_path, base_name, rect_id); 
  
        for joint_id = 1:14
            % Convert predictions for un-cropped image scale for evaluation
            id = convert_joint_order(joint_id);
            
            pred_point_raw = [x_preds_raw(joint_id), y_preds_raw(joint_id)];
            [x_pred, y_pred]= convert_trans_and_scale(pred_point_raw, mid_point, scale);
            %[x_pred, y_pred]= convert_trans_and_scale_validation(pred_point_raw, mid_point, scale);
            
            % Draw joints on the original image
            plot(x_pred, y_pred,'o');
        end
    end
    hold off;
    pause;
end