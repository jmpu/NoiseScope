import os
import scipy
import numpy as np
import random
from sklearn import metrics
import math
import scipy.io
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import matlab.engine
import glob
import mahotas as mt
import random
import argparse

random.seed(6666)
eng = matlab.engine.start_matlab()


def clip_fp(fp):
    '''
    clip values into (0, 255)
    :param fp: A fingerprint
    :return: Clipped fingerprint
    '''
    clipped_fp = np.clip(fp, 0, 1)
    ret_fp = (clipped_fp * 255).astype(int)
    return ret_fp


def extract_haralick_features(image):
    '''
    Extract haralick feature for an image
    :param image: a clipped fingerprint output by clip_fp
    :return: haralick texture feature of the fingerprint
    '''
    textures = mt.features.haralick(image)
    ht_mean = textures.mean(axis=0)
    return ht_mean


def texture_feat_extract(res_folder, res_dim, total_round=5, fp_size=50):
    '''
    This function will 1) create a bunch of fingerprints by randomly sampling 2) will extract texture
    feature from those fingerprints 3) real a feature set
    :param res_folder: noise residual folder of reference set. should only include real image residuals
    :param res_dim: image dimension
    :param total_round: randomly sample for 5 rounds by default
    :param fp_size: each fingerprint is extracted from 50 residuals
    :return: A set of feature of reference fingerprints (real residuals calculated)
    '''
    feat_set = []
    for round in range(0, total_round):
        res = os.listdir(res_folder)
        random.shuffle(res)
        print("There are {} available noise residuals".format(len(res)))
        seg_idxs = [tuple(range(x, x + fp_size)) for x in range(0, len(res) - fp_size, fp_size)]
        for i, seg_idx in enumerate(seg_idxs):
            print("[STATUS] Creating fingerprint {}".format(i))
            res_paths_for_one_fp = list(map(lambda x: res_folder + res[x], seg_idx))
            FP = eng.compute_fp_from_path(res_paths_for_one_fp, res_dim)
            clipped_fp = clip_fp(np.array(FP))
            feat_vector = extract_haralick_features(clipped_fp)
            feat_set.append(feat_vector)
    print('[STATUS] TRAIN feature extraction DONE')
    return feat_set


def compute_pce_with_fingerprint(res_list, fingerprint):
    '''
    For each residual in a list of noise residuals, compute its pce correlation with a fingerpint.
    :param res_list: A list of noise residuals (can be all the test residuals or all the reference residuals)
    :param fingerprint: A fingerprint.
    :return: an array of PCE correlation.
    '''
    ret_pce = eng.compute_pce_with_fingerprint(res_list, matlab.double(fingerprint.tolist()))
    return np.array(ret_pce)


def compute_fp_from_cluster(idxs, res_list, img_dim):
    '''
    compute a fingerprint out of a cluster of residuals by averaging.
    :param idxs: the indexes of a residual cluster
    :param res_list: noise residuals of test set.
    :param img_dim: image/residual dimension.
    :return: A fingerprint.
    '''
    averaged_fp = np.zeros((img_dim, img_dim))
    for idx in idxs:
        fp = scipy.io.loadmat(res_list[idx - 1])  # type: 'dict'
        averaged_fp += fp['Noise'] / len(idxs)
    return np.array(averaged_fp)


def compute_cluster_fake_purity(cluster_with_img_idx, ground_truth):
    '''
    Compute the percentage of fake images/residuals in a cluster
    :param cluster_with_img_idx: A list of residual clusters. Each cluster is a tuple, which includes residual indexes.
    :param ground_truth: ground truth labels of the test set
    :return: a percentage
    '''
    cluster_idx_minus = list(map(lambda x: x - 1, cluster_with_img_idx))
    fake_pos = np.where(np.array(ground_truth) == 1)
    fake_purity = len(set(fake_pos[0]).intersection(set(cluster_idx_minus))) / len(cluster_with_img_idx)
    return fake_purity


def compute_confusion_matrix(ground_truth, label):
    '''
    compute detection performance given ground truth label and prediction label
    :param ground_truth: ground truth label of the test set
    :param label: prediction label of the test set.
    :return: metric scores
    '''
    tn, fp, fn, tp = confusion_matrix(ground_truth, label).ravel()
    conf_matrix = (tn, fp, fn, tp)
    metric_scores = {
        "accuracy": metrics.accuracy_score(ground_truth, label),
        "precision": metrics.precision_score(ground_truth, label),
        "recall": metrics.recall_score(ground_truth, label),
        "f1_score": metrics.f1_score(ground_truth, label)
    }
    return conf_matrix, metric_scores


def save_fingerprint_imgs(res_folder, img_dim, num_res=150):
    '''
    To visualize fingerprint.
    :param res_folder: the path to noise residuals of images from a specific camera/GAN model
    :param img_dim: image/noise dimension
    :param num_res: the number of noise residuals used for creating a fingerprint
    :return:
    '''
    files = glob.glob(res_folder + '*.mat')[:num_res]
    eng.visualize_fingerprint(files, img_dim, './StyleGAN_bedroom_FP.png')
    print('fingerprint saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_res_dir', default='/alldata/residuals/StyleGAN_bedroom/', help='PATH to directory that contains noise residuals from a GAN model')
    args = parser.parse_args()
    save_fingerprint_imgs(args.gan_res_dir, 256)
