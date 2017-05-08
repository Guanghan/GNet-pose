function evaluatePCP(predidxs,evalMode,bSave)
% implementation of strict PCP measure,
% as defined in [Ferrari et al., CVPR'08]

fprintf('evaluatePCP\n');

if (nargin < 3)
    bSave = true;
end
range = 0.5;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end
% load ground truth
assert(strcmp(evalMode,'PC') || strcmp(evalMode,'OC'));
load(['./joints-' evalMode '.mat'],'joints');
tableTex = cell(length(predidxs)+1,1);

fprintf('evalMode: %s\n',evalMode);

for i = 1:length(predidxs);
    
    % load predictions
    p = getExpParams(predidxs(i));
    load(p.predFilename,'pred');
    disp(p.predFilename);
    disp(i);
    
    % compute distance to ground truth joints
    dist = getDistPCP(pred,joints(1:2,:,1001:2000));
    
    % compute PCK
    pcp = computePCP(dist,range);
    
    % plot results
    [row, header] = genTablePCP(pcp(end,:),p.name);
    tableTex{1} = header;
    tableTex{i+1} = row;
end

if (bSave)
    fid = fopen([tableDir '/pcp-' evalMode '.tex'],'wt');assert(fid ~= -1);
    for i=1:length(tableTex),fprintf(fid,'%s\n',tableTex{i}); end; fclose(fid);
end

end