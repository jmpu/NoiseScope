function corr_dist = get_pce_dist(res_list, idx_list, img_size)
% This function compute pairwise PCE correlation among a bunch of fingerprints.
% INPUT:
%   res_list      A list of residual paths
%   idx_list      A list of tuples, each tuple includes residual indexes that forms a fingerprint
%   img_size      image dimension
% OUTPUT:
%   corr_dist     A list of PCE values.
idx_len = length(idx_list);
pce_matrix = zeros([idx_len, idx_len]);
corr_dist = [];
fp_cell = cell(1, length(idx_list));

for m = 1:length(idx_list)
    Noise_sum = zeros([img_size, img_size]);
    for n = 1:length(idx_list{m})
        idx = idx_list{m}{n};
        res_file_path = res_list(idx);
        noise = load(char(res_file_path));
        Noise_sum = Noise_sum + double(noise.Noise);
    end
    fp = Noise_sum/length(idx_list{m});
    fp_cell{1, m} = fp;
end

for i = 1: length(fp_cell)
    for j = 1:length(fp_cell)
        if i > j
            fp1 = fp_cell{1, i};
            fp2 = fp_cell{1, j};
            ccorr = crosscorr(fp1,fp2); 
            out = PCE(ccorr);
            pce_matrix(i, j) = out.PCE;
            corr_dist = [corr_dist; out.PCE];
        end
    end
end
