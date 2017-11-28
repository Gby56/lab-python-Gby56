#This lab was done on Colaboratory (colab.research.google.com)


#////////////////////////////////INITIAL SETUP

# -*- coding: utf-8 -*-
#!mkdir images
#!mkdir out

#!wget http://data.teklia.com/Images/HORAE/Latin_13263_btv1b9068094h.tar
#!tar -xvf Latin_13263_btv1b9068094h.tar -C images/
#!ls -lia images/Latin_13263_btv1b9068094h

#!pip install joblib
#!pip install tqdm

#////////////////////////////////RUNNING EVERYTHING &
#!python cluster_images.py --images-dir images/Latin_13263_btv1b9068094h/ --move-images out/

#!tar cvf - out/ - > file.tar

#from google.colab import files
#files.download('file.tar')

#////////////////////////////////UPLOAD TO GOOGLE DRIVE
# Install the PyDrive wrapper & import libraries.
# This only needs to be done once in a notebook.
'''!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials'''

# Authenticate and create the PyDrive client.
# This only needs to be done once in a notebook.
'''auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)'''

# Create & upload a text file.
'''uploaded = drive.CreateFile({'title': 'file.tar'})
uploaded.SetContentFile('file.tar.gz')
uploaded.Upload()
print 'Uploaded file with ID', uploaded.get('id')'''


'''
#UPLOAD SCRIPT HERE TO EXECUTE VIA COMMAND LINE ON SERVER
from google.colab import files

uploaded = files.upload()
with open("cluster_images.py", 'w') as f:
    f.write(uploaded[uploaded.keys()[0]])
'''



# -*- coding: utf-8 -*-
'''
Cluster images based on visual similarity

C. Kermorvant - 2017
'''

import argparse
import logging
import os
import shutil
import multiprocessing
import sys

from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import numpy as np

# default sub-resolution
IMG_FEATURE_SIZE = (12, 16)

# Setup logging
logger = logging.getLogger('cluster_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def extract_features(img):
    """
    Compute the subresolution of an image and return it as a feature vector

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
    :rtype: list of int in [0,255]

    """

    # convert color images to grey level
    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        IMG_FEATURE_SIZE, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]


def copy_to_dir(images, clusters, cluster_dir):
    """
    Move images to a directory according to their cluster name

    :param images: list of image names (path)
    :type images: list of path
    :param clusters: list of cluster values (int), such as given by cluster.labels_, associated to each image
    :type clusters: list
    :param cluster_dir: prefix path where to copy the images is a drectory corresponding to each cluster
    :type images: path
    :return: None
    """

    for img_path, cluster in zip(images, clusters):
        # define the cluster path : for example "CLUSTERS/4" if the image is in cluster 4
        clst_path = os.path.join(cluster_dir, str(cluster))
        # create the directory if it does not exists
        if not os.path.exists(clst_path):
            os.mkdir(clst_path)
        # copy the image into the cluster directory
        shutil.copy(img_path, clst_path)


def get_jpg_files():
    """ Returns a list of jpeg files in the input directory """
    files = []
    print(os.path.dirname(os.path.realpath(__file__)) + "/" + args.images_dir)
    for file in os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/" + args.images_dir):
        if ".jpg" not in file:
            continue
        files.append(os.path.dirname(os.path.realpath(__file__)) + "/" + args.images_dir + file)
    return files


def image_loader(image):
    """" Extract the features from the image"""
    my_image = Image.open(image)
    return extract_features(my_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, cluster images and move them to a directory')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--move-images')
    args = parser.parse_args()
    CLUSTER_DIR = ""
    if args.move_images:
        CLUSTER_DIR = args.move_images
        # Clean up
        if os.path.exists(CLUSTER_DIR):
            shutil.rmtree(CLUSTER_DIR)
            logger.info('remove cluster directory %s' % CLUSTER_DIR)
        os.mkdir(CLUSTER_DIR)

    # find all the pages in the directory
    images_path_list = []
    data = []
    if args.images_dir:
        SOURCE_IMG_DIR = args.images_dir
        images_path_list = get_jpg_files()

    if not images_path_list:
        logger.warning("Did not found any jpg image in %s" % args.images_dir)
        sys.exit(0)

    # Multi-core feature extraction.
    data = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(image_loader)(image) for image in tqdm(images_path_list))

    if not data:
        logger.error("Could not extract any feature vector")
        sys.exit(1)


    # convert to np array (default format for scikit-learn)
    X = np.array(data)
    logger.info("Running clustering")


    kmeans_model = KMeans(n_clusters=11, random_state=1).fit(X)

    if not args.images_dir:
        logger.info(msg="Cluster image directory was not specified, exiting")
        sys.exit(1)

    copy_to_dir(images_path_list, kmeans_model.labels_, CLUSTER_DIR)