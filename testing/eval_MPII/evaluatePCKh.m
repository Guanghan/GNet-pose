function evaluatePCKh(predidxs,bSave)
% implementation of PCKh measure,
% as defined in [Andriluka et al., CVPR'14]

fprintf('evaluatePCKh()\n');

if (nargin < 2)
    bSave = true;
end

range = 0:0.01:0.5;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end
plotsDir = './plots'; if (~exist(plotsDir,'dir')), mkdir(plotsDir); end
tableTex = cell(length(predidxs)+1,1);

% load ground truth
p = getExpParams(-1);
load([p.gtDir '/annolist_dataset_v12'],'annolist');  % do not have this file. so the author does not release ground truth?
load([p.gtDir '/mpii_human_pose_v1_u12'],'RELEASE');

annolist_test = annolist(RELEASE.img_train == 0);
% evaluate on the "single person" subset only
single_person_test = RELEASE.single_person(RELEASE.img_train == 0);
% convert to annotation list with a single pose per entry
[annolist_test_flat, single_person_test_flat] = flatten_annolist(annolist_test,single_person_test);
% represent ground truth as a matrix 2x14xN_images
gt = annolist2matrix(annolist_test_flat(single_person_test_flat == 1));
% compute head size
headSize = getHeadSizeAll(annolist_test_flat(single_person_test_flat == 1));

pckAll = zeros(length(range),16,length(predidxs));

for i = 1:length(predidxs);

    % load predictions
    p = getExpParams(predidxs(i));
    load([fileparts(p.predFilename) '/pred_keypoints_mpii' ],'pred');
    assert(length(annolist_test) == length(pred));
    pred_flat = flatten_annolist(pred,single_person_test);
    pred = annolist2matrix(pred_flat(single_person_test_flat == 1));
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
