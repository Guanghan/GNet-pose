function [x_pred, y_pred]= convert_trans_and_scale_validation(pred_point_raw, scale_in_cpp, mid_point)
   % 1. The input heatmap is 256 by 256, which has been [resized, padded, cropped] from original image
   %    This script finds position in the original image.
   
   cropped_size = 368;
   
   x_normalized = pred_point_raw(1);
   y_normalized = pred_point_raw(2);
   
   x_pred = double(x_normalized * cropped_size/256.0 - cropped_size/2.0)/ scale_in_cpp + mid_point(1);
   y_pred = double(y_normalized * cropped_size/256.0 - cropped_size/2.0)/ scale_in_cpp + mid_point(2);
  
end