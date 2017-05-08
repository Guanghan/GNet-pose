
% keypointsAll - struct of 14x2 predictions: rankle rknee rhip lhip lknee lankle neck
%                                            head, rwrist, relbow, rshoulder, lshoulder, 
%                                            lelbow, lwrist
% 
% annolist - annotation list with single annotated pose per image. In case
% of multiple people per image, split into several images, each with single
% pose

% evaluate PCKh
sc = 1;
factor = threshold;
display(sc);
display(factor);
isMatchAll = eval_match_keypoints_gt(keypointsAll, annolist, sc, factor);
eval_plot_pck(isMatchAll,1:14);
