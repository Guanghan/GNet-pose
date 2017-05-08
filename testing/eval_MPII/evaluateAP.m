function evaluateAP(predidxs)
% implementation of AP measure,
% as defined in [Pishchulin et al., arXiv'15]

fprintf('evaluateAP()\n');

thresh = 0.5;

tableTex = cell(length(predidxs)+1,1);

% load ground truth
p = getExpParams(-1);
load([p.gtDir '/annolist_dataset_v12'],'annolist');
load([p.gtDir '/mpii_human_pose_v1_u12'],'RELEASE');
% load groups of people
load('./groups_v12','groups');
bTrain = false;
% obtain testing multi-person images and rectangles
[imgidxs_multi_test,rectidxs_multi_test] = getMultiPersonGroups(groups,RELEASE, bTrain);
% all testing images
annolist_test = annolist(RELEASE.img_train == 0);
% multi-person images
annolist_test_multi = annolist_test(imgidxs_multi_test);
% multi-person rectangles
for imgidx = 1:length(annolist_test_multi)
    annolist_test_multi(imgidx).annorect = annolist_test_multi(imgidx).annorect(rectidxs_multi_test{imgidx});
end

apAll = zeros(15,length(predidxs));
markerAll = repmat({'-'},length(predidxs),1);

tableDir = [fileparts(p.predFilename) '/latex']; if (~exist(tableDir,'dir')), mkdir(tableDir); end

for i = 1:length(predidxs);
    
    p = getExpParams(predidxs(i));
    % load predictions
    load([fileparts(p.predFilename) '/pred_keypoints_mpii_multi' ],'pred');
    assert(length(annolist_test_multi) == length(pred));
    % assign predicted poses to GT poses
    [scoresAll, labelsAll, nGTall] = assignGTmulti(pred,annolist_test_multi,thresh(end));
    % compute average precision (AP) per part
    ap = zeros(size(nGTall,1)+1,1);
    for j = 1:size(nGTall,1)
      scores = []; labels = [];
      for imgidx = 1:length(annolist_test_multi)
        scores = [scores; scoresAll{j}{imgidx}];
        labels = [labels; labelsAll{j}{imgidx}];
      end
      % compute precision/recall
      [precision,recall] = getRPC(scores,labels,sum(nGTall(j,:)));
      % compute AP
      ap(j) = VOCap(recall,precision)*100;
    end
    % compute mean AP
    ap(end) = mean(ap(1:end-1));
    columnNames = p.partNames;
    save([fileparts(p.predFilename) '/apAll'],'ap','columnNames');
    % plot results
    [row, header] = genTableAP(ap,p.name);
    tableTex{1} = header;
    tableTex{i+1} = row;
    apAll(:,i) = ap;
end

fid = fopen([tableDir '/ap.tex'],'wt');assert(fid ~= -1);
for i=1:length(tableTex),fprintf(fid,'%s',tableTex{i}); end; fclose(fid);

end