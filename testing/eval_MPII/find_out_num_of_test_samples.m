% Author: Guanghan Ning, March 2017
clear all;
clc;
close all;
addpath('../utils_eval_mpii/');
addpath('../eval_MPII/');
mat_folder_path = '../dataset_mpii/results_all_test_cropped/quantitative/';

% Load annotations: train + validation + test
load('../eval_MPII/mpii_human_pose_v1_u12_1','RELEASE');
annolist = RELEASE.annolist;
annolist_test = annolist(RELEASE.img_train == 0);

pred = annolist;
rectidxs = RELEASE.single_person(RELEASE.img_train == 0);
%%
num_of_single_persons= 0;
for imgidx = 1:6908
        % Get rect id. Note the difference between rect_ct and rect_id
        rect_id_set = rectidxs{imgidx, 1};
        num_of_rects = size(rect_id_set, 1);
        
        flag_empty = isempty(rect_id_set);
        if flag_empty == 1
            disp(rect_id_set);
            continue; %skip this empty one
        end
        
        num_of_single_persons = num_of_single_persons + num_of_rects;
end  
        