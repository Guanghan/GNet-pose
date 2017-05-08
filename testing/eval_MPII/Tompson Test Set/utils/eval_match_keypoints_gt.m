function isMatchAll = eval_match_keypoints_gt(keypointsAll, annolist, sc, factor, bHeadSizeFromRect)

% PCKh evaluation - PCK where distances are scaled by head size
% keypointsAll - struct of 14x2 predictions: rankle rknee rhip lhip lknee lankle neck
%                                            head, rwrist, relbow, rshoulder, lshoulder, 
%                                            lelbow, lwrist
% 
% annolist - annotation list with single annotated pose per image. In case
% of multiple people per image, split into several images, each with single
% pose
% sc - rescale the detections
% factor - threshold for PCK evaluation
% extract head size from annotation points

if ischar(annolist)
    annolist = loadannotations(annolist);
end

if (nargin < 3)
    sc = 1;
end

if (nargin < 4)
    factor = 0.5;
end

if (nargin < 5)
    bHeadSizeFromRect = true;
end

[~, ~, keypoints] = util_get_parts();

assert(length(keypointsAll) == length(annolist));
assert(bHeadSizeFromRect == true);
%assert(factor == 0.5);

isMatchAll = nan(length(annolist),length(keypoints));

for imgidx = 1:length(annolist)
    fprintf('.');
    rect = annolist(imgidx).annorect;
    points = rect.annopoints.point;
    if (bHeadSizeFromRect)
        headSize =  util_get_head_size(rect);
    else
        p1 = util_get_annopoint_by_id(points,8);
        p2 = util_get_annopoint_by_id(points,9);
        headSize = norm([p1.x p1.y] - [p2.x, p2.y]);
    end
    
    keypoints_det = keypointsAll{imgidx};

    if (isempty(keypoints_det))
        for kidx = 1:length(keypoints)
            p = util_get_annopoint_by_id(points,keypoints(kidx).pos);
            if (~isempty(p))
                isMatchAll(imgidx,kidx) = 0;
            end
        end
    else
        assert(length(keypoints_det) == length(keypoints));
        for kidx = 1:length(keypoints)
            p = util_get_annopoint_by_id(points,keypoints(kidx).pos);
            det    = 1/sc*keypoints_det(kidx,:);
            
            if (~isempty(p)) 
                gt = [p.x p.y];

                dist = norm(det - gt);
                gtScale = headSize;
                
                bIsGTmatch = is_gt_match_pck(dist, factor, gtScale);
                isMatchAll(imgidx,kidx) = bIsGTmatch;
            end
        end
    end
    if (~mod(imgidx, 100))
        fprintf(' %d/%d\n',imgidx,length(annolist));
    end
end
fprintf('\ndone\n');

    function res = is_gt_match_pck(dist, factor, gtLen)
        res = dist <= gtLen*factor;
    end

end
