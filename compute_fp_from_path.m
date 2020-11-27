function FP = compute_fp_from_path(res_list, img_size)
% This function compute a fingerprint from a list of noise residuals.
% INPUT:
%   res_list      A list of residual file paths
%   img_size      residual dimension
% OUTPUT:
%   FP      The fingerprint by simply averaging the noise residuals.
Noise_sum = zeros([img_size, img_size]);
for i = 1 : length(res_list)
    img_file_path = res_list(i);
    noise = load(char(img_file_path));
    % if you feed in a list of image files rather than a list of residuals
    % noise = NoiseExtractFromImage(char(img_file), 2.0, 0, 1);
    Noise_sum = Noise_sum + double(noise.Noise);
end
FP = Noise_sum/length(res_list);

