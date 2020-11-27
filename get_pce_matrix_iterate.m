function matrix = get_pce_matrix_iterate(res_list, cluster_list_with_image_idx, merged_cluster_num, img_dim, dir, iter)
% This function compute PCE matrix which includes pairwise PCE correlation among all current clusters.
% INPUT:
%   res_list      A list of residual paths
%   cluster_list_with_image_idx      A list of tuples, each tuple includes residual indexes that forms a fingerprint
%   merged_cluster_num   number of merged clusters from the last round of clustering.
%   img_dim image dimension
%   dir: folder saves matrix
%   iter: the merging round.
% OUTPUT:
%   matrix     A matrix of PCE values.

custom_cluster = parcluster();
custom_cluster.JobStorageLocation = dir;
parpool(custom_cluster, 6);
matrix = zeros([merged_cluster_num, length(cluster_list_with_image_idx)]);
fp_cell = cell(1, length(cluster_list_with_image_idx));

for m = 1:length(cluster_list_with_image_idx)
    Noise_sum = zeros([img_dim, img_dim]);
    for n = 1:length(cluster_list_with_image_idx{m})
        idx = cluster_list_with_image_idx{m}{n};
        img_file_path = res_list(idx);
        Residual = load(char(img_file_path));
        Noise_sum = Noise_sum + double(Residual.Noise);
    end
    fp = Noise_sum/length(cluster_list_with_image_idx{m});
    fp_path = strcat(dir, 'iter', int2str(iter), '_fp', int2str(m), '.mat');
    save(fp_path, 'fp');
    fp_cell{1, m} = fp_path;
end


num_cluster = length(cluster_list_with_image_idx);
parfor i = 1: merged_cluster_num
    fp1_path = fp_cell{1, num_cluster - merged_cluster_num + i};
    fp1 = load(char(fp1_path));
    updates = zeros(1, length(fp_cell));
    for j = 1: num_cluster - merged_cluster_num + i - 1
        fp2_path = fp_cell{1, j};
        fp2 = load(char(fp2_path));
        ccorr = crosscorr(fp1.fp,fp2.fp);
        out = PCE(ccorr);
        updates(j) = out.PCE;
    end
    matrix(i, :) = updates;
end
save(strcat(dir, int2str(iter), '_partial.mat'), 'matrix');
delete(strcat(dir, 'iter', int2str(iter), '*.mat'));
p = gcp;
delete(p);
