function [x_pred, y_pred]= convert_scale_cropped(pred_point_raw, org_img_size, scale_in_cpp)
   % 1. The input heatmap is 256 by 256, which has been [resized, padded, cropped] from original image
   %    This script finds position in the original image.

   cropped_size = 368;

   org_img_wid = org_img_size(1);
   org_img_ht = org_img_size(2);

   x_normalized = pred_point_raw(1);
   y_normalized = pred_point_raw(2);

   x_pred = double(x_normalized * cropped_size/256.0 - cropped_size/2.0)/ scale_in_cpp + org_img_wid/2.0;
   y_pred = double(y_normalized * cropped_size/256.0 - cropped_size/2.0)/ scale_in_cpp + org_img_ht/2.0;

end
