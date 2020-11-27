import matlab.engine
import numpy as np
from sklearn.svm import OneClassSVM
import os
import glob
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn import metrics
import argparse
import logging
import random

random.seed(6666)
eng = matlab.engine.start_matlab()

def get_color_feat(img_paths):
    '''
    Extract color feature for all the images
    :param img_paths: image paths
    :return: a list of feature
    '''
    all_feats = []
    for path in img_paths:
        if os.path.exists(path):
            feat = eng.gan_img_detection_fea(path)
            all_feats.append(feat)
        else:
            logging.info('this file does not exist. HELP!')
    logging.info("The length of all the feat", len(all_feats))
    return all_feats


def train_CSD_SVM(args):
    '''
    Train a SVM outlier detector using real images
    :param real_img_dir: A directory contains real images
    :param svm_model_path: A path for saving trained model
    :return:
    '''
    train_paths = list(map(lambda x: args.real_img_dir + x, os.listdir(args.real_img_dir)))
    logging.info("Training file paths: {}".format(len(train_paths)))
    train_feat = get_color_feat(train_paths)
    train_feat = np.squeeze(train_feat, axis=1)
    y_true = [1] * np.shape(train_feat)[0]
    # train SVM
    parameters = {'gamma': [0.001, 0.0001, 1 / 588, 0.01, 0.1]}
    svm_model = OneClassSVM(nu=0.1, kernel="rbf")
    clf = GridSearchCV(svm_model, parameters, cv=5, scoring='accuracy')
    clf.fit(train_feat, y_true)
    logging.info(clf.best_estimator_.get_params())
    # save the model
    joblib.dump(clf.best_estimator_, args.svm_model_path)
    logging.info('model saved')


def test_CSD_SVM(args):
    '''
    Test the trained CSD-SVM model
    :param real_img_dir: Directory of real images
    :param fake_img_dir: Directory of fake images
    :param svm_model_path: Trained model
    :return: Detection performance
    '''
    real_paths = list(map(lambda x: args.real_img_dir + x, random.sample(os.listdir(args.real_img_dir), args.num_real)))
    real_feat = get_color_feat(real_paths)
    fake_paths = list(map(lambda x: args.fake_img_dir + x, random.sample(os.listdir(args.fake_img_dir), args.num_fake)))
    fake_feat = get_color_feat(fake_paths)

    test_feat = real_feat + fake_feat
    test_label = [1] * len(real_feat) + [-1] * len(fake_feat)

    test_feat = np.squeeze(test_feat, axis=1)

    svm_model = joblib.load(args.svm_model_path)
    pred_labels = svm_model.predict(test_feat)
    metric_scores = {
        "accuracy": metrics.accuracy_score(test_label, pred_labels),
        "precision": metrics.precision_score(test_label, pred_labels),
        "recall": metrics.recall_score(test_label, pred_labels),
        "f1_score": metrics.f1_score(test_label, pred_labels)
    }
    logging.info("F1 score", metric_scores['f1_score'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_img_dir', default='/rdata/jiameng/DeepLens/alldata/original_imgs/flickr_winter/refer/', help='path to training image dir, which includes real images only')
    parser.add_argument('--real_img_dir', default='/rdata/jiameng/DeepLens/alldata/original_imgs/flickr_winter/real/', help='path to real image dir for testing')
    parser.add_argument('--fake_img_dir', default='/rdata/jiameng/DeepLens/alldata/original_imgs/CycleGAN_winter/fake/', help='path to fake image dir for testing')
    parser.add_argument('--num_real', default=500, help='The number of real images in the test set')
    parser.add_argument('--num_fake', default=500, help='The number of fake images in the test set')
    parser.add_argument('--svm_model_path', default='./winter_example.pkl', help='The path that trained SVM model will be saved')
    args = parser.parse_args()
    logging.basicConfig(filename='./csd_svm.log', filemode='w', level=logging.INFO, format='%(levelname)s:%(message)s')
    train_CSD_SVM(args)
    test_CSD_SVM(args)
