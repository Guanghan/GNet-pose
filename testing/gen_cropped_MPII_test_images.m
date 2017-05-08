function gen_cropped_MPII_test_images()
% Author: Guanghan Ning, Feb 2017
clear all;
clc;

target_dist = 1.45;
boxsize = 368;

% Get [input images paths] and [output images path]
mpii_img_folder_path = '../dataset/MPI/images/';
img_output_folder = 'dataset_mpii/images_test_cropped/qualitative/';

% Load annotations
load('eval_MPII/mpii_Tompson_test', 'annolist');
load('eval_MPII/mpii_Tompson_test', 'keypointsAll');
load('eval_MPII/mpii_Tompson_test', 'RELEASE_img_index');
load('eval_MPII/mpii_Tompson_test', 'RELEASE_person_index');

annolist_test = annolist;
num_test_imgs = size(annolist_test, 2);

load('eval_MPII/mpii_human_pose_v1_u12_1','RELEASE');
annolist_all = RELEASE.annolist;

%% Start from here if everything already read from mat file.
for img_id = 1:num_test_imgs

    img_index = RELEASE_img_index(img_id);

    % (1) Deal with rect ids

        % Get rect id. Note the difference between rect_ct and rect_id
        rect_id = RELEASE_person_index(img_id);

        % Find scale and mid-points in order to convert prediction
        x_mid = annolist_test(img_id).annorect.objpos.x;
        y_mid = annolist_test(img_id).annorect.objpos.y;
        objpos_org = [x_mid, y_mid];
        scale_provided = annolist_test(img_id).annorect.scale;

        % (2) Deal with images
        img_name = annolist_all(img_index).image.name;
        img_path = strcat(mpii_img_folder_path, img_name);
        if ~exist(img_path, 'file')
            continue
        end

        img = imread(img_path);

        scale_in_cpp = target_dist/scale_provided;
        center_s = objpos_org * scale_in_cpp;
        fprintf('scale: %f\n', scale_in_cpp);

        img = imresize(img, scale_in_cpp);
        disp(size(img));
        [img, pad] = padAround(img, boxsize, center_s);
        disp(size(img));

        [pathstr, name, ext] = fileparts(img_path);
        img_output_path = strcat(img_output_folder, name, '_', num2str(rect_id), ext);
        imwrite(img, img_output_path);

        output_scale(img_id) = scale_in_cpp;
end
save('scale_in_cpp_MPII_test.mat', 'output_scale');


function [img_padded, pad] = padAround(img, boxsize, center)
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right

    pad_up = repmat(img(1,:,:)*0, [pad(1) 1 1])+128;
    img_padded = [pad_up; img];
    pad_left = repmat(img_padded(:,1,:)*0, [1 pad(2) 1])+128;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:)*0, [pad(3) 1 1])+128;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:)*0, [1 pad(4) 1])+128;
    img_padded = [img_padded pad_right];

    center = center + [max(0,pad(2)) max(0,pad(1))];
    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed


function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
