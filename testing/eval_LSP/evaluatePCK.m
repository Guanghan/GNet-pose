function evaluatePCK(predidxs,evalMode,bSave)
% implementation of PCK measure,
% as defined in [Sapp&Taskar, CVPR'13].
% torso height: ||left_shoulder - right hip||

fprintf('evaluatePCK\n');

if (nargin < 3)
    bSave = true;
end

range = 0:0.01:0.2;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end

% load ground truth
assert(strcmp(evalMode,'PC') || strcmp(evalMode,'OC'));
load(['./joints-' evalMode '.mat'],'joints');
tableTex = cell(length(predidxs)+1,1);

fprintf('evalMode: %s\n',evalMode);

pckAll = zeros(length(range),15,length(predidxs));

for i = 1:length(predidxs);
    
    % load predictions
    p = getExpParams(predidxs(i));
    load(p.predFilename,'pred');
    disp(p.predFilename);
    disp(i);
    
    % compute distance to ground truth joints
    dist = getDistPCK(pred,joints(1:2,:,1001:2000));
    
    % compute PCK
    pck = computePCK(dist,range);
    
    % plot results
    [row, header] = genTablePCK(pck(end,:),p.name);
    tableTex{1} = header;
    tableTex{i+1} = row;
    
    pckAll(:,:,i) = pck;
    
    auc = area_under_curve(scale01(range),pck(:,end));
%     plot(range,pck(:,end),'color',p.colorName,'LineStyle','-','LineWidth',3);
    fprintf('%s, AUC: %1.1f\n',p.name,auc);
end

if (bSave)
    fid = fopen([tableDir '/pck-' evalMode '.tex'],'wt');assert(fid ~= -1);
    for i=1:length(tableTex),fprintf(fid,'%s\n',tableTex{i}); end; fclose(fid);
end

% plot curves
plotCurve(squeeze(pckAll(:,end,:)),range,predidxs,['PCK total, LSP ' evalMode],['./plots/pck-total-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[1 6],:),2)),range,predidxs,['PCK ankle, LSP ' evalMode],['./plots/pck-ankle-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[2 5],:),2)),range,predidxs,['PCK knee, LSP ' evalMode],['./plots/pck-knee-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[3 4],:),2)),range,predidxs,['PCK hip, LSP ' evalMode],['./plots/pck-hip-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[7 12],:),2)),range,predidxs,['PCK wrist, LSP ' evalMode],['./plots/pck-wrist-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[8 11],:),2)),range,predidxs,['PCK elbow, LSP ' evalMode],['./plots/pck-elbow-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[9 10],:),2)),range,predidxs,['PCK shoulder, LSP ' evalMode],['./plots/pck-shoulder-lsp-' evalMode],bSave);
%plotCurve(squeeze(mean(pckAll(:,[13 14],:),2)),range,predidxs,['PCK head, LSP ' evalMode],['./plots/pck-head-lsp-' evalMode],bSave);
% title(['PCK, LSP ' evalMode]);
% legend(legendName,'Location','NorthWest');
% set(gca,'YLim',[0 100],'xtick',range(1:2:end),'ytick',0:10:100);
% xlabel('Normalized distance');
% ylabel('Detection rate, %');
% grid on;

 if (bSave)
     print(gcf, '-dpng', ['./plots/pck-lsp-' evalMode '.png']);
     printpdf(['./plots/pck-lsp-' evalMode '.pdf']);
     savefig(['./plots/pck-lsp-' evalMode '.fig']);
     fid = fopen([tableDir '/pck-' evalMode '.tex'],'wt');assert(fid ~= -1);
     for i=1:length(tableTex),fprintf(fid,'%s\n',tableTex{i}); end; fclose(fid);
 end

end