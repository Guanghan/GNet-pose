function dist = getDistPCK(pred,gt)

assert(size(pred,1) == size(gt,1) && size(pred,2) == size(gt,2) && size(pred,3) == size(gt,3));

dist = nan(1,size(pred,2),size(pred,3));

for imgidx = 1:size(pred,3)
    
    % torso diameter
    refDist = norm(gt(:,10,imgidx) - gt(:,3,imgidx));
    
    % distance to gt joints
    dist(1,:,imgidx) = sqrt(sum((pred(:,:,imgidx) - gt(:,:,imgidx)).^2,1))./refDist;

end