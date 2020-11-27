import numpy as np
import scipy.io
import os
import matlab.engine
import glob
import mahotas as mt
import pickle
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neighbors import LocalOutlierFactor
from sklearn.externals import joblib
from utils_noisescope import *
import logging
import argparse
import random

random.seed(6666)
eng = matlab.engine.start_matlab()

# NoiseScope step 1: noise extractor
def noise_calculator(filepath, res_dir):
    '''
    Noise residual extractor for a single image
    :param filepath: An image file
    :param res_dir: A folder that saves image residual
    :return: the noise residual
    '''
    Noisex = eng.NoiseExtractFromImage(filepath, 2.0, 0, 1, res_dir)
    return Noisex


def save_residual(img_dir, res_dir):
    '''
    NoiseScope Step 1: noise residual extractor
    :param img_dir: a folder that contains images
    :param res_dir: a folder that saves extracted noise residuals.
    :return: none
    '''
    if res_dir[-1] != '/': res_dir = res_dir + '/'
    if img_dir[-1] != '/': img_dir = img_dir + '/'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    glob_files = glob.glob(img_dir + '*.JPEG') + glob.glob(img_dir + '*.jpg') + glob.glob(img_dir + '*.png')  # only grab png files or jpeg files
    print(len(glob_files))
    for i, img in enumerate(glob_files):
        noise_calculator(img, res_dir)


# calibrate threshold for T_merge
def correlation_between_real_fps(refer_res_dir, dim, fp_size=20, percentile=99.5):
    '''

    :param refer_res_dir: The folder contains noise residuals of real images in the reference set
    :param dim: image size.
    :param fp_size: number of fingerprint used to calibrate the T_merge
    :return:
    '''
    all_refer_res = os.listdir(refer_res_dir) # By default, the list of files returned by os.listdir() is in arbitrary order.
    refer_res = list(map(lambda x: os.path.join(refer_res_dir, x), all_refer_res[: 40 * fp_size]))
    assert len(refer_res) % fp_size == 0
    logging.info('{} real fingerprints are used to calibrate T_merge'.format(len(refer_res) / fp_size))

    seg_idxs = [tuple(range(x, x + fp_size)) for x in range(1, len(refer_res), fp_size)]
    corr_dist = eng.get_pce_dist(refer_res, seg_idxs, dim)
    logging.info(corr_dist)
    merge_thre = np.percentile(corr_dist, percentile)
    logging.info('Merging threshold is calibrated as: {}'.format(round(merge_thre, 2)))


def fingerprint_outlier_detector(res_dir, res_dim, outlier_model_path, data_exist=False, train_feat_path=''):
    '''

    :param res_dir:
    :param outlier_model_path:
    :param data_exist:
    :param train_feat_path:
    :return:
    '''
    if not data_exist:
        logging.info("Creating Training Feature set")
        train_feat = texture_feat_extract(res_dir, res_dim)
    else:
        assert os.path.isfile(train_feat_path)
        logging.info("[STATUS] Loading Training Feature set")
        with open(train_feat_path, 'rb') as f:
            train_feat = pickle.load(f)

    # create the classifier
    logging.info("[STATUS] Creating the classifier..")
    clf = LocalOutlierFactor(n_neighbors=30, novelty=True, contamination=0.00001)
    # fit the training data and labels
    logging.info("[STATUS] Fitting data/label to model..")
    clf.fit(train_feat)
    joblib.dump(clf, outlier_model_path)
    return


def prep_pipeline(args):
    # NoiseScope step 1: noise residual extractor.
    # This step can be skipped if you already have noise residuals ready.
    # Convert all real images into noise residuals, and save them in real_res_dir.
    # save_residual(args.real_img_dir, args.real_res_dir)
    # Convert all fake images into noise residuals, and save them in fake_res_dir.
    # save_residual(args.fake_img_dir, args.fake_res_dir)
    # Convert all reference images into noise residuals, and save them in refer_res_dir.
    # save_residual(args.refer_img_dir, args.refer_res_dir)
    # This step can be skipped if you already have noise residuals ready.

    # calibrate T_merge threshold with the help of reference set. T_merge will be used in NoiseScope step 2
    correlation_between_real_fps(args.refer_res_dir, args.img_dim)
    # train a fingerprint outlier detector with the help of reference set. Outlier detector will be used in NoiseScope step 3.
    # fingerprint_outlier_detector(args.refer_res_dir, args.img_dim, args.fingerprint_classifier_path)


if __name__ == '__main__':
    # please resize image to square before apply noiseScope.
    # Prepare three folder of images:
    # Test images for detection are sampled from real_img_dir and fake_img_dir.
    # refer_res_dir is a folder of real images used as reference by NoiseScope. (It has no overlap with images in real_img_dir)
    # This prep_pipeline will prepare steps before applying NoiseScope.
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_img_dir', default='', help='path to REAL image dir')
    parser.add_argument('--real_res_dir', default='', help='Specify the path of REAL noise residual dir which the extracted residuals will be saved to.')
    parser.add_argument('--fake_img_dir', default='', help='path to FAKE image dir')
    parser.add_argument('--fake_res_dir', default='', help='Specify the path of FAKE noise residual dir which the extracted residuals will be saved to.')

    parser.add_argument('--refer_img_dir', default='', help='path to REFERENCE image dir')
    parser.add_argument('--refer_res_dir', default='/rdata/jiameng/DeepLens/alldata/residuals/flickr_winter/refer/', help='Specify the path of REFERENCE noise residual dir which the extracted residuals will be saved to.')

    parser.add_argument('--img_dim', type=int, help='image dimension.', default=256)
    parser.add_argument('--fingerprint_classifier_path', type=str,
                        help='Specify the path of fingerprint outlier detector to save.', default='./out_reproduce_classifier.pkl')
    args = parser.parse_args()
    logging.basicConfig(filename='./out_reproduce.log', filemode='w', level=logging.DEBUG, format='%(levelname)s:%(message)s')
    prep_pipeline(args)

