function [x_pred, y_pred]= convert_trans_and_scale(pred_point_raw, mid_point, scale)
   % 1. The input heatmap is 248 by 248, which is rescaled from a crop, introducing
   %    a scale
   % 2. The crop comes from the original image, and introduce a translation
   % Summary: This script is to do the backwards mapping, back to the original position.
   % (1) First, find the position in the cropped image
   % (2) Second, find the position in the original image

   % Use (scale) for the scale. [180, 280] is current value.
   % Consult [gen_cropped_test_images.m]. Make sure use same scale.
   % Officially, scale is w.r.t. 200 px height
   human_bbox_wid = 180 * scale;
   human_bbox_ht = 280 * scale;
   
   % Prepare [x_min, y_min] for translation
   x_min = mid_point(1) - human_bbox_wid/2.0;
   y_min = mid_point(2) - human_bbox_ht/2.0;

   % Based on translation, scale, and output, find input
   [x_pred, y_pred] = find_pos_for_org_image(pred_point_raw, human_bbox_wid, human_bbox_ht, x_min, y_min);
end


function [x0, y0] = find_pos_for_org_image(pos_248, human_bbox_wid, human_bbox_ht, x_min, y_min)
    [x1, y1] = find_pos_for_cropped_image(pos_248, human_bbox_wid, human_bbox_ht);
    
    x0 = x1 + x_min;
    y0 = y1 + y_min;
    
    pos_org = [x0, y0];
end


function [x1, y1] = find_pos_for_cropped_image(pos_248, human_bbox_wid, human_bbox_ht)
   x2 = pos_248(1);
   y2 = pos_248(2);
   
   x1 = int64((x2 * human_bbox_wid)/248.0);
   y1 = int64((y2 * human_bbox_ht)/248.0);
   
   pos_cropped = [x1, y1];
end

