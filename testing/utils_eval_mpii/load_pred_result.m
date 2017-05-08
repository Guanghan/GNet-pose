function [x_preds, y_preds] = load_pred_result(mat_folder_path, img_basename, rect_id)
   mat_name = strcat(img_basename, '_', num2str(rect_id), '.mat');
   mat_path = strcat(mat_folder_path, '/', mat_name);
   disp(mat_name);
   pred_struct = load(mat_path);
   x_preds = pred_struct.joints(:, 2);
   y_preds = pred_struct.joints(:, 1);
end

