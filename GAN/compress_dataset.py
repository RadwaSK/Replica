import os
import numpy as np
import cv2 as cv

def resize_image(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    Resize image with given width or height, keeping aspect ration

    :param image: Input image
    :param width: Desired width
    :param height: Desired height
    :param inter: Type of desired interpolation

    :return: Resized image
    """

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

if __name__ == '__main__':
    subject_name = input("Enter subject name: ")
    abs_path = input("Enter absolute path, -1 if none: ")
    if abs_path == '-1':
        abs_path = '/media/inno/500gigaExtra/Replica/datasets/'
    abs_path += subject_name
    train_images_file_names = os.listdir(abs_path + '/train/train_img/')
    train_poses_file_names = os.listdir(abs_path + '/train/train_label/')
    test_images_file_names = os.listdir(abs_path + '/val/test_img/')
    test_poses_file_names = os.listdir(abs_path + '/val/test_label/')

    if not os.path.exists(abs_path + '_compressed'):
        os.makedirs(abs_path + '_compressed')
        os.makedirs(abs_path + '_compressed' + '/train/train_img/')
        os.makedirs(abs_path + '_compressed' + '/train/train_label/')
        os.makedirs(abs_path + '_compressed' + '/val/test_img/')
        os.makedirs(abs_path + '_compressed' + '/val/test_label/')

    n = len(train_images_file_names)
    for i in range(n):
        name = abs_path + '/train/train_img/' + train_images_file_names[i]
        img = cv.imread(name)
        img = resize_image(img, width=512)
        cv.imwrite(abs_path + '_compressed/train/train_img/' + train_images_file_names[i], img)

        img = cv.imread(abs_path + '/train/train_label/' + train_poses_file_names[i])
        img = resize_image(img, width=512)
        cv.imwrite(abs_path + '_compressed/train/train_label/' + train_poses_file_names[i], img)

    n = len(test_poses_file_names)
    for i in range(n):
        img = cv.imread(abs_path + '/val/test_img/' + test_images_file_names[i])
        img = resize_image(img, width=512)
        cv.imwrite(abs_path + '_compressed/val/test_img/' + test_images_file_names[i], img)

        img = cv.imread(abs_path + '/val/test_label/' + test_poses_file_names[i])
        img = resize_image(img, width=512)
        cv.imwrite(abs_path + '_compressed/val/test_label/' + test_poses_file_names[i], img)