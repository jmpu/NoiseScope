function matrix= get_pce_matrix_noise_average(res_list, cluster_list_with_image_idx, dir, iter, img_dim)
% This function compute PCE matrix which includes pairwise PCE correlation among all noise residuals (before merging/clustering).
% INPUT:
%   res_list      A list of residual paths
%   cluster_list_with_image_idx      A list of tuples, each tuple includes residual indexes that forms a fingerprint
%   dir: folder saves matrix
%   iter: the merging round.
%   img_dim image dimension
% OUTPUT:
%   matrix     A matrix of PCE values.
    custom_cluster = parcluster();
    custom_cluster.JobStorageLocation = dir;
    parpool(custom_cluster, 6);
    matrix = zeros([length(cluster_list_with_image_idx), length(cluster_list_with_image_idx)]);
    parfor m = 1:length(cluster_list_with_image_idx)
        Noise_sum = zeros([img_dim, img_dim]);
        for n = 1:length(cluster_list_with_image_idx{m})
            idx = cluster_list_with_image_idx{m}{n};
            residual_path = res_list(idx);
        end
        fp_cell{1, m} = residual_path;
    end

    num_clusters = length(cluster_list_with_image_idx);
    disp(num_clusters)
    tic
    parfor i = 1: num_clusters
        updates = zeros(1, num_clusters);
        fp1 = load(char(fp_cell{1, i}))
        for j = 1: num_clusters
            if i > j
                fp2 = load(char(fp_cell{1, j}))
                ccorr = crosscorr(fp1.Noise, fp2.Noise);
                out = PCE(ccorr);
                updates(j) = out.PCE;
            end
        matrix(i, :) = updates;
        end
    end
    toc
    save(strcat(dir, int2str(iter), '.mat'), 'matrix');
    p = gcp;
    delete(p);
end