clearvars; clc; close all;
addpath('./utils');

load('data/detections.mat');

threshold = 0.5;
SC_BIAS = 0.6; % THIS IS DEFINED IN util_get_head_size.m


if ~strcmp(type, 'test')
  run('evaluate.m')
  
  load('data/detections_our_format.mat');
  head = find(ismember(dataset_joints, 'head'));
  
  lsho = find(ismember(dataset_joints, 'lsho'));
  lelb = find(ismember(dataset_joints, 'lelb'));
  lwri = find(ismember(dataset_joints, 'lwri'));
  lhip = find(ismember(dataset_joints, 'lhip'));
  lkne = find(ismember(dataset_joints, 'lkne'));
  lank = find(ismember(dataset_joints, 'lank'));
  
  rsho = find(ismember(dataset_joints, 'rsho'));
  relb = find(ismember(dataset_joints, 'relb'));
  rwri = find(ismember(dataset_joints, 'rwri'));
  rhip = find(ismember(dataset_joints, 'rhip'));
  rkne = find(ismember(dataset_joints, 'rkne'));
  rank = find(ismember(dataset_joints, 'rank'));
  
  % Calculate PCKh again for a few joints just to make sure our evaluation
  % matches Leonid's...
  jnt_visible = 1 - jnt_missing;
  uv_err = pos_pred_src - pos_gt_src;
  uv_err = sqrt(sum(uv_err .* uv_err, 2));
  headsizes = headboxes_src(2,:,:) - headboxes_src(1,:,:);
  headsizes = sqrt(sum(headsizes .* headsizes, 2));
  headsizes = headsizes * SC_BIAS;
  scaled_uv_err = squeeze(uv_err ./ repmat(headsizes, size(uv_err, 1), 1, 1));
  
  % Zero the contribution of joints that are missing
  scaled_uv_err = scaled_uv_err .* jnt_visible;
  jnt_count = squeeze(sum(jnt_visible, 2));
  less_than_threshold = (scaled_uv_err < threshold) .* jnt_visible;
  PCKh = 100 * squeeze(sum(less_than_threshold, 2)) ./ jnt_count;
  
  fprintf(' Head & Shoulder & Elbow & Wrist & Hip & Knee  & Ankle\n');
  fprintf('%1.1f & %1.1f & %1.1f & %1.1f & %1.1f & %1.1f & %1.1f\n',...
    PCKh(head), (PCKh(lsho)+PCKh(rsho))/2, (PCKh(lelb)+PCKh(relb))/2,...
    (PCKh(lwri)+PCKh(rwri))/2, (PCKh(lhip)+PCKh(rhip))/2, ...
    (PCKh(lkne)+PCKh(rkne))/2, (PCKh(lank)+PCKh(rank))/2);
  
else
  disp('Nothing to do for test set (GT is held out)');
end

