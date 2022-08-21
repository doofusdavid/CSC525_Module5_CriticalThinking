"""
Module 5 - Critical Thinking - Option 1
David Edwards
CSC525 - Principles of Machine Learning
Dr. Issac Gang
8/20/2022
"""

import imageio.v3 as iio
import imgaug.augmenters as iaa
import os
import random
import cv2


def hflip(image):
    aug = iaa.Fliplr(1.0)
    return aug.augment_image(image)


def vflip(image):
    aug = iaa.Flipud(1.0)
    return aug.augment_image(image)


def channel_shuffle(image):
    aug = iaa.ChannelShuffle(1.0)
    return aug.augment_image(image)


def rotate_img(image):
    aug = iaa.Affine(rotate=(-30, 30))
    return aug.augment_image(image)


def add_noise(image):
    aug = iaa.AdditiveGaussianNoise(10, 40)
    return aug.augment_image(image)


def elastic_transform(image):
    aug = iaa.ElasticTransformation(alpha=10.0, sigma=1.0)
    return aug.augment_image(image)


process_functions = [hflip, vflip, channel_shuffle, rotate_img, add_noise, elastic_transform]

file_count = 0
for file in os.listdir("."):
    if file.endswith(".png"):
        file_count += 1
        image = iio.imread(file)
        image = cv2.resize(image, (256, 256))
        iio.imwrite("./augmented/resized_" + file, image)
        # apply two random transformations, five times per image
        for i in range(5):
            for _ in range(2):
                image = random.choice(process_functions)(image)
            new_file = "./augmented/aug_{}_{}".format(str(i), file)
            iio.imwrite(new_file, image)
print("{} files augmented".format(file_count))
