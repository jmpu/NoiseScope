function FP = compute_fp_from_path(img_list, img_size, fp_path)
% This function compute a fingerprint from a list of noise residuals.
% INPUT:
%   res_list      A list of residual file paths
%   img_size      residual dimension
%   fp_path       the path saves the computed fingerprint
% OUTPUT:
%   FP      The fingerprint by simply averaging the noise residuals.
Noise_sum = zeros([img_size, img_size]);
for i = 1 : length(img_list)
    img_file_path = img_list(i);
    noise = load(char(img_file_path));
    %noise = NoiseExtractFromImage(char(img_file_path), 2.0, 0, 1);
    Noise_sum = Noise_sum + double(noise.Noise);
end
FP = Noise_sum/length(img_list);
imwrite(FP, fp_path);

