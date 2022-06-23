# NoiseScope
This repo is the implementation of NoiseScope for [NoiseScope: Detecting Deepfake Images in a Blind Setting](https://jmpu.github.io/files/acsac2020-57-noisescope.pdf) published in ACSAC 2020.

We propose a blind detection approach called NoiseScope for discovering GAN images among real images. NoiseScope requires no a priori access to fake images for training, and demonstrably generalizes better than supervised detection schemes. 

## Datasets

We evaluated 11 datasets covering 4 GANs. Here we provide 5 images datasets covering 4 GANs: [StyleGAN-Face1](https://drive.google.com/file/d/1-tujp1z6t3RvQdcMzMoyZ0CW3zHUcxqQ/view?usp=sharing), [StyleGAN-Face2](https://drive.google.com/file/d/1-o3ELnu_1LjflehtW4IirxrIpm4sg1A0/view?usp=sharing), [PGGAN-Tower](https://drive.google.com/file/d/1-cwfpOfYbIq8Upj_y8wO2nFMPTETImS0/view?usp=sharing), [CycleGAN-Winter](https://drive.google.com/file/d/1-Zt3HJtUzN4HI7TBglKuhH8CvyyGluH5/view?usp=sharing), [BigGAN-DogHV](https://drive.google.com/file/d/1-pzi9K4sXCr-UPutpovdOhggtDoitAop/view?usp=sharing).

To run on those datasets, pretrained fingerprint classifiers are [here](https://drive.google.com/file/d/101OQkjkHhwrHzjcvSRSAB4vw9OlfdOuB/view?usp=sharing). The estimated `T_merge` values for those datasets are noted in `run_NoiseScope.sh`
## NoiseScope Usage

The code is based on Python 3 and Matlab. We borrow the [Camera Fingerprint - Matlab implementation](http://dde.binghamton.edu/download/camera_fingerprint/) to extract noise residuals.

1. Install the dependencies as below:
- First find your matlab installation folder, and install the matlab engine for python with: `python -u setup.py install`. Then install requests library with `pip install requests`, then do `pip install -r requirements.txt`.

2. Compilation: 

- Run commands in `./Filter/compile.m`

   _According to [Camera Fingerprint - Matlab implementation](http://dde.binghamton.edu/download/camera_fingerprint/): "A collection of supportive functions is in \Functions while noise extraction is in \Filter directory. Wavelet transform functions mdwt.dll and midwt.dll were compiled under 32-bit Windows operating system. The source code provided by Rice University is included. Run mex command if you need to compile it under different OS (see compile.m)."_


3. Apply NoiseScope for detection.

 - **First --- Data preparation.** you need to prepare three data folders: real folder containing real images, reference folder containing real images, fake folder with fake images from GANs you want to detect. Images in real folder and reference folder usually share the same data distribution and the same type of image content. Ideally, we recommend to collect no less than 1000 real images as reference data.

  - **Second --- Data preprocessing and parameter calibration.** Use `prep_steps.py` to: 1) Extract noise residual for above images (used in NoiseScope Step 1), 2) Calibrate the merging threshold T_merge which will be used in NoiseScope Step 2, 3) Train a fingerprint classifier which will be used in NoiseScope Step 3. If you use noise residuals in the datasets released by us, you can skip the first step of noise residual extraction in `prep_steps.py`.

```python
python prep_steps.py 
--real_img_dir: The path to REAL image dir
--real_res_dir: Specify the path of REAL noise residual dir which the extracted residuals will be saved to.
--fake_img_dir: The path to FAKE image dir.
--fake_res_dir: Specify the path of FAKE noise residual dir which the extracted residuals will be saved to.
--refer_img_dir: The path to REFERENCE image dir.
--refer_res_dir: Specify the path of REFERENCE noise residual dir which the extracted residuals will be saved to.
--img_dim: image dimension.
--fingerprint_outlier_model_path:Specify the path of fingerprint outlier detector to save.
# You will get an estimated merging threshold and a trained outlier detector.
  ``` 

  - **Third --- Running the main clustering algorithm.** Specify data paths and detection configurations obtained from the preparation procedure in `run_NoiseScope.sh`.

```python
# specify below parameters in run_NoiseScope.sh
 python noisescope_clustering.py
--real_res_dir: The path to path to REAL noise residual dir 
--fake_res_dir: The path to FAKE noise residual dir
--refer_res_dir: The path to REFERENCE noise residual dir
--num_real: The number of real images in the test set
--num_fake: The number of fake images in the test set
--img_dim: Image dimension.
--outlier_model_path: The path to pre-trained fingerprint outlier detector
--result_dir: Specify the folder which saves log file and some matrix files produced in the middle
--pce_thre: T_merging threshold estimated
``` 
## To visualize fingerprints

   - To visualize a fingerprint shown in Figure 2 in the paper for a GAN, specify the data path and run 

`python utils_noisescope.py --gan_res_dir [PATH to directory that contains noise residuals from a GAN model]`


##  Compare NoiseScope with [CSD-SVM](https://arxiv.org/abs/1808.07276)

   Train a outlier detector and test on the testing data by run `python util_CSD_svm.py`.

```python
python util_CSD_svm.py
--train_img_dir: path to training image dir, which includes real images only
--real_img_dir: path to real image dir for testing
--fake_img_dir: path to fake image dir for testing
--num_real: The number of real images in the test set
--num_fake: The number of fake images in the test set
--svm_model_path: The path that trained SVM model will be saved.
``` 

## Citation
```
@inproceedings{pu-2020-acsac,
 author	= {Jiameng Pu and Neal Mangaokar and Bolun Wang and Chandan K. Reddy and Bimal Viswanath},
title = {NoiseScope: Detecting Deepfake Images in a Blind Setting},
booktitle = {Proceedings of the Annual Computer Security Applications Conference (ACSAC)},
year = {2020}
}


```
**If you have any questions, please contact <jmpu@vt.edu>.**
