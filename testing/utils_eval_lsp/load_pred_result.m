function [x_preds, y_preds] = load_pred_result(mat_folder_path, img_basename)
   % Load pred mat file for each test image from LSP dataset.
   mat_name = strcat(img_basename, '.mat');
   mat_path = strcat(mat_folder_path, '/', mat_name);
   %disp(mat_path);
   pred_struct = load(mat_path);
   x_preds = pred_struct.joints(:, 2);
   y_preds = pred_struct.joints(:, 1);
   %disp(x_preds);
end