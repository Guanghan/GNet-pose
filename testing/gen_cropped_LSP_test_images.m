function gen_cropped_LSP_test_images()
% Author: Guanghan Ning, Feb 2017
        addpath('../training/Annotations/util/jsonlab/');
        crop_size = 368;

        % in cpp: real scale = param_.target_dist()/meta.scale_self = (41/35)/scale_input
        targetDist = 41/35; % in caffe cpp file 41/35
        oriTrTe = load('../dataset/LEEDS/lsp_dataset/joints.mat');
        extTrain = load('../dataset/LEEDS/lspet_dataset/joints.mat');

        flag_showJoints = 0;
        flag_showScaledImg = 0;

        % in LEEDS:
        % 1  Right ankle
        % 2  Right knee
        % 3  Right hip
        % 4  Left hip
        % 5  Left knee
        % 6  Left ankle
        % 7  Right wrist
        % 8  Right elbow
        % 9  Right shoulder
        % 10 Left shoulder
        % 11 Left elbow
        % 12 Left wrist
        % 13 Neck
        % 14 Head top
        % 15,16 DUMMY
        % We want to comply to MPII: (1 - r ankle, 2 - r knee, 3 - r hip, 4 - l hip, 5 - l knee, 6 - l ankle, ..
        %                             7 - pelvis, 8 - thorax, 9 - upper neck, 10 - head top,
        %                             11 - r wrist, 12 - r elbow, 13 - r shoulder, 14 - l shoulder, 15 - l elbow, 16 - l wrist)
        ordering = [1 2 3, 4 5 6, 15 16, 13 14, 7 8 9, 10 11 12]; % should follow MPI 16 parts..?
        oriTrTe.joints(:,[15 16],:) = 0;
        oriTrTe.joints = oriTrTe.joints(:,ordering,:);
        oriTrTe.joints(3,:,:) = 1 - oriTrTe.joints(3,:,:);
        oriTrTe.joints = permute(oriTrTe.joints, [2 1 3]);

        extTrain.joints([15 16],:,:) = 0;
        extTrain.joints = extTrain.joints(ordering,:,:);

        count = 1;

        path = {'lspet_dataset/images/im%05d.jpg', 'lsp_dataset/images/im%04d.jpg'};
        local_path = {'../dataset/LEEDS/lspet_dataset/images/im%05d.jpg', '../dataset/LEEDS/lsp_dataset/images/im%04d.jpg'};

        num_image = [10000, 2000];

        for dataset = 1:2
            for im = 1001:num_image(dataset)
                path_this = sprintf(local_path{dataset}, im);
                if exist(path_this, 'file') == 0
                    continue
                end
                disp(im);
                disp(count);

                % trivial stuff for LEEDS
                joint_all(count).dataset = 'LEEDS';
                joint_all(count).isValidation = 0;
                joint_all(count).img_paths = sprintf(path{dataset}, im);
                joint_all(count).numOtherPeople = 0;
                joint_all(count).annolist_index = count;
                joint_all(count).people_index = 1;
                % joints and w, h
                if(dataset == 1)
                    joint_this = extTrain.joints(:,:,count);
                else
                    joint_this = oriTrTe.joints(:,:,im);
                end

                [h,w,~] = size(imread(path_this));

                joint_all(count).img_width = w;
                joint_all(count).img_height = h;
                joint_all(count).joint_self = joint_this;
                % infer objpos
                invisible = (joint_all(count).joint_self(:,3) == 0);
                if(dataset == 1) %lspet is not tightly cropped
                    joint_all(count).objpos(1) = (min(joint_all(count).joint_self(~invisible, 1)) + max(joint_all(count).joint_self(~invisible, 1))) / 2;
                    joint_all(count).objpos(2) = (min(joint_all(count).joint_self(~invisible, 2)) + max(joint_all(count).joint_self(~invisible, 2))) / 2;
                else
                    joint_all(count).objpos(1) = w/2;
                    joint_all(count).objpos(2) = h/2;
                end
                % visualize
                if (flag_showJoints == 1)
                 figure(1); clf; imshow(path_this);
                 hold on;
                 plot(joint_all(count).joint_self([1 2 3], 1), joint_all(count).joint_self([1 2 3], 2), 'wx', 'Linewidth', 3);
                 plot(joint_all(count).joint_self([4 5 6], 1), joint_all(count).joint_self([4 5 6], 2), 'bx', 'Linewidth', 3);
                 plot(joint_all(count).joint_self([9 10], 1), joint_all(count).joint_self([9 10], 2), 'gx', 'Linewidth', 3);
                 plot(joint_all(count).joint_self([11 12 13], 1), joint_all(count).joint_self([11 12 13], 2), 'mx', 'Linewidth', 3);
                 plot(joint_all(count).joint_self([14 15 16], 1), joint_all(count).joint_self([14 15 16], 2), 'yx', 'Linewidth', 3);
                 plot(joint_all(count).joint_self(invisible, 1), joint_all(count).joint_self(invisible, 2), 'rx');
                 plot(joint_all(count).objpos(1), joint_all(count).objpos(2), 'cs');
                 pause;
                end
                % increase counter and display info
                count = count + 1;
                fprintf('processing %s\n', path_this);
            end
        end

        joint_all = insertMPILikeScale(joint_all, targetDist);


         for i = 1:length(joint_all)
             path = ['../dataset/LEEDS/', joint_all(i).img_paths];
             %figure(1); clf;
             img = imread(path);
             scale_in_cpp = targetDist/joint_all(i).scale_provided;
             disp(size(img));
             img = imresize(img, scale_in_cpp);
             disp(size(img));
             img = [128*ones(400,size(img,2),3); img];
             img = [128*ones(size(img,1),400,3), img];
             img = [img; 128*ones(400,size(img,2),3)];
             img = [img, 128*ones(size(img,1),400,3)];

             fprintf('scale: %f\n', scale_in_cpp);
             objpos = joint_all(i).objpos * scale_in_cpp + 400;

             if (flag_showScaledImg == 1)
             imshow(img);
             hold on;
             plot(objpos(1), objpos(2), 'cs');
             line([objpos(1)-368/2, objpos(1)-368/2], [objpos(2)+368/2, objpos(2)-368/2], 'lineWidth', 3);
             line([objpos(1)+368/2, objpos(1)-368/2], [objpos(2)+368/2, objpos(2)+368/2], 'lineWidth', 3);
             line([objpos(1)+368/2, objpos(1)+368/2], [objpos(2)-368/2, objpos(2)+368/2], 'lineWidth', 3);
             line([objpos(1)-368/2, objpos(1)+368/2], [objpos(2)-368/2, objpos(2)-368/2], 'lineWidth', 3);
             pause;
             end

             disp(i);
             xmin = objpos(1) - crop_size/2;
             ymin = objpos(2) - crop_size/2;
             wid = crop_size;
             ht = crop_size;
             rect = [xmin ymin wid ht];
             img_cropped = imcrop(img, rect);
             img_out_path = sprintf('dataset_lsp/images_cropped/qualitative/im%04d.jpg', 1000+i);
             imwrite(img_cropped, img_out_path);

             if (i == 1)
                disp(scale_in_cpp);
             end

             output_scale(i) = scale_in_cpp;
         end
         save('scale_in_cpp_LSP.mat', 'output_scale');


function joint_all = insertMPILikeScale(joint_all, targetDist)
    % calculate scales for each image first
    joints = cat(3, joint_all.joint_self);
    joints([7 8],:,:) = [];
    pa = [2 3 7, 5 4 7, 8 0, 10 11 7, 13 12 7];
    x = permute(joints(:,1,:), [3 1 2]);
    y = permute(joints(:,2,:), [3 1 2]);
    vis = permute(joints(:,3,:), [3 1 2]);
    validLimb = 1:14-1;

    x_diff = x(:, [1:7,9:14]) - x(:, pa([1:7,9:14]));
    y_diff = y(:, [1:7,9:14]) - y(:, pa([1:7,9:14]));
    limb_vis = vis(:, [1:7,9:14]) .* vis(:, pa([1:7,9:14]));
    l = sqrt(x_diff.^2 + y_diff.^2);

    for p = 1:14-1 % for each limb. reference: 7th limb, which is 7 to pa(7) (neck to head)
        valid_compare = limb_vis(:,7) .* limb_vis(:,p);
        ratio = l(valid_compare==1, p) ./ l(valid_compare==1, 7);
        r(p) = median(ratio(~isnan(ratio), 1));
    end

    numFiles = size(x_diff, 1);
    all_scales = zeros(numFiles, 1);

    boxSize = 368;
    psize = 64;
    nSqueezed = 0;

    for file = 1:numFiles %numFiles
        l_update = l(file, validLimb) ./ r(validLimb);
        l_update = l_update(limb_vis(file,:)==1);
        distToObserve = quantile(l_update, 0.75);
        scale_in_lmdb = distToObserve/35; % can't get too small. 35 is a magic number to balance to MPI
        scale_in_cpp = targetDist/scale_in_lmdb; % can't get too large to be cropped

        visibleParts = joints(:, 3, file);
        visibleParts = joints(visibleParts==1, 1:2, file);
        x_range = max(visibleParts(:,1)) - min(visibleParts(:,1));
        y_range = max(visibleParts(:,2)) - min(visibleParts(:,2));
        scale_x_ub = (boxSize - psize)/x_range;
        scale_y_ub = (boxSize - psize)/y_range;

        scale_shrink = min(min(scale_x_ub, scale_y_ub), scale_in_cpp);

        if scale_shrink ~= scale_in_cpp
            nSqueezed = nSqueezed + 1;
            fprintf('img %d: scale = %f %f %f shrink %d\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub), nSqueezed);
        else
            fprintf('img %d: scale = %f %f %f\n', file, scale_in_cpp, scale_shrink, min(scale_x_ub, scale_y_ub));
        end

        joint_all(file).scale_provided = targetDist/scale_shrink; % back to lmdb unit
    end

    fprintf('total %d squeezed!\n', nSqueezed);
