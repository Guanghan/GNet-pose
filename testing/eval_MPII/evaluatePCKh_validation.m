function evaluatePCKh_validation(predidxs,bSave)
% implementation of PCKh measure,
% as defined in [Andriluka et al., CVPR'14]

fprintf('evaluatePCKh_validation()\n');

if (nargin < 2)
    bSave = true;
end

range = 0:0.01:0.5;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end
plotsDir = './plots'; if (~exist(plotsDir,'dir')), mkdir(plotsDir); end
tableTex = cell(length(predidxs)+1,1);

% load ground truth
load('mpii_Tompson_validation', 'annolist');
load('mpii_Tompson_validation', 'keypointsAll');
load('mpii_Tompson_validation', 'RELEASE_img_index');
load('mpii_Tompson_validation', 'RELEASE_person_index');

% represent ground truth as a matrix 2x14xN_images
% need to make gt this format: [2 x 14 x 2958]
%num_test_imgs = size(keypointsAll, 2);
%gt = zeros(2, 14, num_test_imgs);
%for img_id = 1:num_test_imgs
%    for joint_id = 1:14 %joint_id in our joint order
%        gt(1, joint_id, img_id) = keypointsAll{1, img_id}(joint_id, 1); 
%        gt(2, joint_id, img_id) = keypointsAll{1, img_id}(joint_id, 2);
%    end
%end

% represent ground truth as a matrix 2x14xN_images
gt = annolist2matrix(annolist);

% compute head size
headSize = getHeadSizeAll(annolist);

pckAll = zeros(length(range),16,length(predidxs));

for i = 1:length(predidxs);

    % load predictions
    p = getExpParams(predidxs(i));
    %load(fileparts(p.predFilename), 'pred_validation');
    disp(p.predFilename);
    load(p.predFilename, 'pred_validation');
    
    pred = pred_validation; 
    assert(length(annolist) == length(pred));
    
    % only gt is allowed to have NaN
    pred(isnan(pred)) = inf;

    % compute distance to ground truth joints
    dist = getDistPCKh(pred,gt,headSize);

    % compute PCKh
    pck = computePCK(dist,range);

    % plot results
    [row, header] = genTablePCK(pck(end,:),p.name);
    tableTex{1} = header;
    tableTex{i+1} = row;

    pckAll(:,:,i) = pck;

    auc = area_under_curve(scale01(range),pck(:,end));
    fprintf('%s, AUC: %1.1f\n',p.name,auc);
end

if (bSave)
    fid = fopen([tableDir '/pckh.tex'],'wt');assert(fid ~= -1);
    for i=1:length(tableTex),fprintf(fid,'%s\n',tableTex{i}); end; fclose(fid);
end

% plot curves
plotCurve(squeeze(pckAll(:,end,:)),range,predidxs,'PCKh total, MPII',[plotsDir '/pckh-total-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[1 6],:),2)),range,predidxs,'PCKh ankle, MPII',[plotsDir '/pckh-ankle-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[2 5],:),2)),range,predidxs,'PCKh knee, MPII',[plotsDir '/pckh-knee-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[3 4],:),2)),range,predidxs,'PCKh hip, MPII',[plotsDir '/pckh-hip-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[7 12],:),2)),range,predidxs,'PCKh wrist, MPII',[plotsDir '/pckh-wrist-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[8 11],:),2)),range,predidxs,'PCKh elbow, MPII',[plotsDir '/pckh-elbow-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[9 10],:),2)),range,predidxs,'PCKh shoulder, MPII',[plotsDir '/pckh-shoulder-mpii'],bSave,range(1:5:end));
plotCurve(squeeze(mean(pckAll(:,[13 14],:),2)),range,predidxs,'PCKh head, MPII',[plotsDir '/pckh-head-mpii'],bSave,range(1:5:end));

end
