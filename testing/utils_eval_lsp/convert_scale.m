function [x_pred, y_pred]= convert_scale(pred_point_raw, org_img_size)
   % 1. The input heatmap is 248 by 248, which is rescaled from original image, 
   %    introducing a scale difference. This script finds position in the
   %    original image.
   org_img_wid = org_img_size(1);
   org_img_ht = org_img_size(2);
   
   x_normalized = pred_point_raw(1);
   y_normalized = pred_point_raw(2);
   
   x_pred = double(x_normalized * org_img_wid)/248.0;
   y_pred = double(y_normalized * org_img_ht)/248.0;
  
   
   if org_img_size(1) == 248 && org_img_size(2)==248
       x_pred = pred_point_raw(1);
       y_pred = pred_point_raw(2);
   end
end
