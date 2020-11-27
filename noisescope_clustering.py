import os
import numpy as np
import random
from sklearn.utils import shuffle
import scipy.io
import matlab.engine
import time
import glob
import argparse
from utils_noisescope import *
import logging
import joblib

random.seed(6666)
eng = matlab.engine.start_matlab()


def extract_fingerpint_via_clustering(all_res_paths, ground_truth_label,
                                      thre_pce,
                                      cluster_list_with_image_idx,
                                      iter_round, img_dim, outlier_model_path, result_dir,
                                      reduce_matrix=None, merged_cluster=None):
    '''
    Fingerprint Step 2 + 3.
    :param all_res_paths: noise residuals of the test set.
    :param ground_truth_label: gound truth labels for the test set.
    :param thre_pce: T merge calibrated using function 'correlation_between_real_fps' in pipeline.py
    :param cluster_list_with_image_idx: A list of residual clusters. Each cluster is a tuple, which includes residual indexes.
    :param iter_round: clustering/merging iteration round
    :param img_dim: image/residual dimension
    :param outlier_model_path: fingerprint outlier detector
    :param result_dir: save log, middle products like .mat files
    :param logfile: log file
    :param reduce_matrix: previous pair-wise correlation reused for this round of merging iteration
    :param merged_cluster: Newly merged clusters from the last merging step
    :return: ret_fake_cluster_list: A list of fake (model) clusters flagged; ret_cluster_list_with_image_idx: residual indexs in the flagged clusters
    '''
    logging.info("++++++++++PERFORM THE NEXT MERGING ITERATION++++++++++++\n")
    logging.info('Currently, there are {} clusters\n'.format(len(
        cluster_list_with_image_idx)))  # cluster_list_with_image_idx show the latest cluster distribution and clusters
    for cluster_with_img_idx in cluster_list_with_image_idx:
        if len(cluster_with_img_idx) > 10:
            fake_purity = compute_cluster_fake_purity(cluster_with_img_idx, ground_truth_label)
            logging.info(
                'This cluster has {} images with a fake purity: {} \n'.format(len(cluster_with_img_idx), fake_purity))
    num_cluster = len(cluster_list_with_image_idx)
    ### calculate PCE matrix ###
    if iter_round > 0:

        pce_matrix = np.full((num_cluster, num_cluster), 0, dtype=float)
        pce_matrix[0:num_cluster - len(merged_cluster), 0: num_cluster - len(merged_cluster)] = reduce_matrix  # 98, 98
        eng.get_pce_matrix_iterate(all_res_paths, cluster_list_with_image_idx, len(merged_cluster), img_dim,
                                   result_dir,
                                   iter_round)
        new_pce_matrix = scipy.io.loadmat(result_dir + '{}_partial.mat'.format(iter_round))
        pce_matrix[num_cluster - len(merged_cluster):, :] = np.array(new_pce_matrix['matrix'])

    else:
        t1 = time.time()
        eng.get_pce_matrix_noise_average(all_res_paths, cluster_list_with_image_idx, result_dir, iter_round,
                                         img_dim)
        t2 = time.time()
        logging.info('The first iteration takes {} seconds. \n'.format(t2 - t1))
        pce_matrix = scipy.io.loadmat(result_dir + '{}.mat'.format(iter_round))
        pce_matrix = np.array(pce_matrix['matrix'])

    large_pce_pos_array = np.where(pce_matrix > thre_pce)
    x_axis_idx = large_pce_pos_array[0]
    y_axis_idx = large_pce_pos_array[1]

    logging.info("{} pairs in the matrix is larger than the threshold. \n".format(len(list(x_axis_idx))))

    # return cases for early stopping
    sorted_cluster_list_with_image_idx = sorted(cluster_list_with_image_idx, key=len, reverse=True)
    # if len(sorted_cluster_list_with_image_idx[0]) > 200:  # if we have a big cluster >200, we test it
    if len(sorted_cluster_list_with_image_idx[
               0]) > 150:  # if we have a big cluster > 150, we start the early stopping strategy
        feed_list = []
        for idx_tuple in sorted_cluster_list_with_image_idx:
            if len(idx_tuple) > 50:  # pick cluster size [50, X)
                feed_list.append(idx_tuple)
            else:
                break
        # return feed_list, tuple_tree_dict, cluster_list_with_image_idx # for skipping

        fake_cluster_list, fake_flagged = fingerprint_classifier(feed_list, all_res_paths,
                                                                 outlier_model_path, img_dim)
        if fake_flagged:
            logging.info(
                "We detected suspicious fake clusters, NoiseScope will perform fingerprint classifier next.")
            return fake_cluster_list, cluster_list_with_image_idx
        else:
            logging.info(
                "Available candidate clusters are not recognized outliers, NoiseScope continues to do clustering.")

    # another return case, when there is no more high correlated pairs
    if len(list(x_axis_idx)) == 0:
        fake_cluster_list, fake_flagged = fingerprint_classifier(sorted_cluster_list_with_image_idx, all_res_paths,
                                                                 outlier_model_path, img_dim)
        if fake_flagged:
            return fake_cluster_list, cluster_list_with_image_idx
        else:
            logging.info("No fake clusters are flagged, NoiseScope will stop the detection.")
            return fake_cluster_list, cluster_list_with_image_idx

    # confirm how many pairs can be merged
    idx_pairs = list(zip(x_axis_idx, y_axis_idx))  # idx_pairs includes all pair positions

    idx_pairs_with_pce = list(map(lambda x: x + (pce_matrix[x[0], x[1]],), idx_pairs))
    sorted_idx_pairs_with_pce = sorted(idx_pairs_with_pce, key=lambda x: x[2], reverse=True)

    idx_pair_for_merge = []
    delete_idxs = []
    while len(sorted_idx_pairs_with_pce) > 0:  # which means still having pairs to merge
        x_idx_max_pce = sorted_idx_pairs_with_pce[0][0]
        y_idx_max_pce = sorted_idx_pairs_with_pce[0][1]
        assert pce_matrix[x_idx_max_pce][y_idx_max_pce] == sorted_idx_pairs_with_pce[0][2]
        idx_pair_for_merge.append((x_idx_max_pce, y_idx_max_pce))
        logging.info(
            'Maximum pce value from current idx pairs is: {}\n'.format(pce_matrix[x_idx_max_pce][y_idx_max_pce]))

        delete_idxs.append(x_idx_max_pce)
        delete_idxs.append(y_idx_max_pce)

        sorted_idx_pairs_with_pce[:] = [idx_pair for idx_pair in sorted_idx_pairs_with_pce if
                                        (x_idx_max_pce not in idx_pair) and (y_idx_max_pce not in idx_pair)]

    ### merging rules ###
    merge_clusters_set = set([])  # contain merged tuples that should be added
    delete_clusters_set = set([])  # contain tuples that need to be deleted
    for idx_pair in idx_pair_for_merge:
        # record all the clusters need to be deleted from cluster_list_with_image_idx
        delete_clusters_set.add(cluster_list_with_image_idx[idx_pair[0]])
        delete_clusters_set.add(cluster_list_with_image_idx[idx_pair[1]])

        # record all the merged cluster need to be added into cluster_list_with_image_idx
        merge_tuple = cluster_list_with_image_idx[idx_pair[0]] + cluster_list_with_image_idx[idx_pair[1]]
        merge_clusters_set.add(merge_tuple)

    # here we remove clusters in delete_clusters_set
    for delete_tuple in delete_clusters_set:
        cluster_list_with_image_idx.remove(delete_tuple)
    # here we add merged clusters in all_merge_set
    for merge_tuple in merge_clusters_set:
        cluster_list_with_image_idx.append(merge_tuple)

    pce_values_for_next_iter = []
    for i in range(0, num_cluster):
        if i in delete_idxs:
            continue
        for j in range(0, num_cluster):
            if j in delete_idxs:
                continue
            pce_values_for_next_iter.append(pce_matrix[i, j])

    pce_matrix = np.reshape(pce_values_for_next_iter, (num_cluster - len(delete_idxs), num_cluster - len(delete_idxs)))

    ret_fake_cluster_list, ret_cluster_list_with_image_idx = extract_fingerpint_via_clustering(all_res_paths,
                                                                                               ground_truth_label,
                                                                                               thre_pce,
                                                                                               cluster_list_with_image_idx,
                                                                                               iter_round + 1,
                                                                                               img_dim,
                                                                                               outlier_model_path,
                                                                                               result_dir,
                                                                                               pce_matrix,
                                                                                               merge_clusters_set)
    return ret_fake_cluster_list, ret_cluster_list_with_image_idx


def fake_image_detector(fake_cluster_list, test_res_paths, ground_truth, img_dim, refer_dir):
    '''
    NoiseScope step 4.
    :param fake_cluster_list: A list of fake clusters. Each cluster includes all the residual indexes.
    :param test_res_paths: noise residual paths for test set.
    :param ground_truth: Ground truth label for the test residuals.
    :param img_dim: image/residual size
    :param logfile: log file
    :param refer_dir: reference dir
    :return: detection F1 score
    '''
    if len(fake_cluster_list) == 0:
        logging.info('No model fingerprint found! The detection will stop here! \n')
        return
    refer_res_paths = glob.glob(refer_dir + '*.mat')
    test_max_pce = []
    refer_max_pce = []
    all_test_pce = []
    all_refer_pce = []
    cluster_stat = []
    single_cluster_f1_scores = []
    for i, fake_cluster in enumerate(fake_cluster_list):
        logging.info('This fake cluster includes residual id: {}. \n'.format(fake_cluster))
        # adjust the index, because in matlab, index starts from 1.
        fake_cluster_idx_minus = list(map(lambda x: x - 1, fake_cluster))
        fake_pos = np.where(np.array(ground_truth) == 1)
        fake_purity = len(set(fake_pos[0]).intersection(set(fake_cluster_idx_minus))) / len(fake_cluster)
        cluster_stat.append((len(fake_cluster), fake_purity))
        logging.info('This cluster has a fake purity of {}. \n'.format(fake_purity))
        logging.info('This cluster has image samples{} \n'.format(len(fake_cluster)))
        model_fingerprint = compute_fp_from_cluster(fake_cluster, test_res_paths, img_dim)
        logging.info('The shape of fake fingerprint: {}. \n'.format(np.shape(model_fingerprint)))
        test_pce_corr = compute_pce_with_fingerprint(test_res_paths, model_fingerprint)
        refer_pce_corr = compute_pce_with_fingerprint(refer_res_paths, model_fingerprint)
        all_test_pce.append(test_pce_corr[0])
        all_refer_pce.append(refer_pce_corr[0])
        if i == 0:
            test_max_pce = test_pce_corr[0]
            refer_max_pce = refer_pce_corr[0]
        else:
            test_max_pce = list(map(lambda x, y: max(x, y), test_max_pce, test_pce_corr[0]))
            refer_max_pce = list(map(lambda x, y: max(x, y), refer_max_pce, refer_pce_corr[0]))

    calibrate_thres = np.percentile(refer_max_pce, 99.5)
    logging.info('Calibrated PCE threshold for fake image detector, {} \n'.format(calibrate_thres))
    label = list(map(lambda x: 1 if x > calibrate_thres else 0, test_max_pce))

    conf_matrix, metric_scores = compute_confusion_matrix(ground_truth, label)
    logging.info("Clustered with PCE threshold: {}. \n".format(calibrate_thres))
    logging.info("TN, FP, FN, TP: {} \n".format(conf_matrix))
    logging.info("+++++++++++++++++++++++++++++++ \n")
    logging.info("Accuracy: {0:.2f}% \n".format(metric_scores["accuracy"] * 100))
    logging.info("Precision: {0:.2f}% \n".format(metric_scores["precision"] * 100))
    logging.info("Recall: {0:.2f}% \n".format(metric_scores["recall"] * 100))
    logging.info("F1 score: {0:.2f}% \n".format(metric_scores["f1_score"] * 100))
    final_f1 = metric_scores["f1_score"]

    for test_pce in all_test_pce:
        label = list(map(lambda x: 1 if x > calibrate_thres else 0, test_pce))

        conf_matrix, metric_scores = compute_confusion_matrix(ground_truth, label)
        logging.info("========Single cluster performance=========\n")
        logging.info("TN, FP, FN, TP: {} \n".format(conf_matrix))
        logging.info("+++++++++++++++++++++++++++++++ \n")
        logging.info("Accuracy: {0:.2f}% \n".format(metric_scores["accuracy"] * 100))
        logging.info("Precision: {0:.2f}% \n".format(metric_scores["precision"] * 100))
        logging.info("Recall: {0:.2f}% \n".format(metric_scores["recall"] * 100))
        logging.info("F1 score: {0:.2f}% \n".format(metric_scores["f1_score"] * 100))
        single_cluster_f1_scores.append(metric_scores["f1_score"])
    return final_f1


def fingerprint_classifier(cluster_list_with_image_idx, res_list, outlier_model_path, img_dim):
    '''
    NoiseScope Step 3: fingerprint classifier
    :param cluster_list_with_image_idx: A list of residual clusters. Each cluster is a tuple, which includes residual indexes.
    :param res_list: Noise residuals of test set.
    :param outlier_model_path: Fingerprint outlier detector, which will flag model fingerprints as outliers
    :param img_dim: image/residual size
    :param logfile: log file
    :return: a list of fake (model) clusters
    '''
    fake_cluster_list = []
    fake_flagged = False

    detection_model = joblib.load(outlier_model_path)
    # cluster_list_with_image_idx = sorted(cluster_list_with_image_idx, key=len, reverse=True)
    for cluster_with_img_idx in cluster_list_with_image_idx:
        if len(cluster_with_img_idx) > 50:  # find the fake set whose size is larger than 50
            sampled_idx = random.sample(cluster_with_img_idx, 50)  # sample cluster_list_with_image_idx

            cluster_fp = compute_fp_from_cluster(sampled_idx, res_list, img_dim)
            clipped_fp = clip_fp(cluster_fp)
            haralick_feat = extract_haralick_features(clipped_fp)

            pred_label = detection_model.predict(np.array(haralick_feat).reshape(1, -1))
            if pred_label == -1:
                fake_cluster_list.append(cluster_with_img_idx)
                logging.info("One fake cluster is flagged, with {} images.\n".format(len(cluster_with_img_idx)))
        else:
            break
    logging.info("{} fake clusters have been flagged.".format(len(fake_cluster_list)))
    if len(fake_cluster_list) > 0: fake_flagged = True
    return fake_cluster_list, fake_flagged


def detection_NoiseScope(args):
    if args.result_dir[-1] != '/': args.result_dir = args.result_dir + '/'
    if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)
    logging.basicConfig(filename='{}detection.log'.format(args.result_dir), filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    real_res_list = random.sample(glob.glob(args.real_res_dir + '/*.mat'), args.num_real)
    fake_res_list = random.sample(glob.glob(args.fake_res_dir + '/*.mat'), args.num_fake)

    all_res_paths = real_res_list + fake_res_list
    ground_truth_label = [0] * len(real_res_list) + [1] * len(fake_res_list)
    shuffle_data = shuffle(list(zip(ground_truth_label, all_res_paths)))
    [ground_truth_label_, all_res_paths_] = zip(*shuffle_data)

    # logfile = open("{}logfile.txt".format(args.result_dir), "w")
    all_res_paths = list(all_res_paths_)
    ground_truth_label = ground_truth_label_

    cluster_list_with_image_idx = [tuple([i]) for i in range(1, len(all_res_paths) + 1)]
    ############ find fake indexs and compute the fake fingerprint ################
    logging.info('Merging threshold: {}\n'.format(args.pce_thre))
    fake_cluster_list, cluster_list_with_image_idx = extract_fingerpint_via_clustering(all_res_paths,
                                                                                       ground_truth_label,
                                                                                       args.pce_thre,
                                                                                       cluster_list_with_image_idx,
                                                                                       0,
                                                                                       args.img_dim,
                                                                                       args.outlier_model_path,
                                                                                       args.result_dir)

    f1_score = fake_image_detector(fake_cluster_list, all_res_paths, ground_truth_label, args.img_dim,
                                   args.refer_res_dir)
    return f1_score


if __name__ == '__main__':
    '''
    We grab 'num_real' samples from 'real_res_dir' and 'num_fake' samples from 'fake_res_dir' 
    specify the 'outlier_model_path' trained from prep_steps.py
    specify 'pce_thre' calibrated from prep_steps.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_res_dir', default='', help='the path to REAL noise residual dir')
    parser.add_argument('--fake_res_dir', default='', help='the path to FAKE noise residual dir')
    parser.add_argument('--refer_res_dir', default='', help='the path to REFERENCE noise residual dir')
    parser.add_argument('--num_real', type=int, help='The number of real images in the test set', default=500)
    parser.add_argument('--num_fake', type=int, help='The number of fake images in the test set', default=500)
    parser.add_argument('--img_dim', type=int, default=256, help='images should be in square shape.')
    parser.add_argument('--outlier_model_path', default='', help='the path to pre-trained fingerprint outlier detector')
    parser.add_argument('--result_dir', default='',
                        help='Specify the folder which saves log file and some matrix files produced in the middle')
    parser.add_argument('--pce_thre', type=float, help='T merging threshold estimated')
    args = parser.parse_args()
    detection_NoiseScope(args)
