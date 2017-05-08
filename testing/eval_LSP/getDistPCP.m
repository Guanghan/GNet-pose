function dist = getDistPCP(pred,gt)

% convert joint annotations into sticks
gt_sticks = keypoints2sticks(gt);
pred_2 = keypoints2sticks(pred);
pred = pred_2;

disp(size(gt_sticks));
disp(size(pred));

assert(size(gt_sticks,2) == 20);
assert(size(pred,1) == size(gt_sticks,1) && size(pred,2) == size(gt_sticks,2) && size(pred,3) == size(gt_sticks,3));

dist = nan(1,size(pred,2),size(pred,3));

for imgidx = 1:size(pred,3)
    for jidx = 1:size(gt_sticks,2)/2
        jidx1 = 2*(jidx-1)+1;
        jidx2 = 2*(jidx-1)+2;
        % distance to gt endpoints
        dist(1,jidx1,imgidx) = norm(gt_sticks(:,jidx1,imgidx) - pred(:,jidx1,imgidx))/norm(gt_sticks(:,jidx1,imgidx) - gt_sticks(:,jidx2,imgidx));
        dist(1,jidx2,imgidx) = norm(gt_sticks(:,jidx2,imgidx) - pred(:,jidx2,imgidx))/norm(gt_sticks(:,jidx1,imgidx) - gt_sticks(:,jidx2,imgidx));
    end
end