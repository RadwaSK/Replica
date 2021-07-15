import os
import numpy as np
import cv2 as cv


def load_dataset(subject_path, st1, en1, st2, en2, train=True, read_images=True, size=(512, 1024)):
    if train:
        if read_images:
            path = subject_path + '/train/train_img/'
        else:
            path = subject_path + '/train/train_label/'
    else:
        if read_images:
            path = subject_path + '/val/test_img/'
        else:
            path = subject_path + '/val/test_label/'
    
    filenames = os.listdir(path)
    dataset_size = len(filenames)
    
    images_list = []
    if train:
        st, en = st1, min(dataset_size, en1)
    else:
        st, en = st2, min(dataset_size, en2)
    if st is None:
        st = 0
    if en == st:
        en = dataset_size
    # enumerate filenames in directory, assume all are images
    for i in range(st, en):
        img = cv.imread(path + filenames[i])
        images_list.append(img)
        
    return np.array(np.array(images_list))


def read_and_save(subject, st1, en1, st2, en2, dataset_path=None):
    if dataset_path is None:
        path = os.getcwd() + '/datasets/' + subject
    else:
        path = dataset_path + subject

    train_images = load_dataset(path, st1, en1, st2, en2, train=True, read_images=True)
    print('Loaded: ', train_images.shape)
    train_filename1 = 'compressed_data/' + subject + '_train_images' + str(st1) + '-' + str(en1) + '.npz'
    if not os.path.exists('compressed_data'):
        os.makedirs('compressed_data')
    np.savez_compressed(train_filename1, train_images)
    del train_images
    print('Saved train dataset', train_filename1)

    train_poses = load_dataset(path, st1, en1, st2, en2, train=True, read_images=False)
    print('Loaded: ', train_poses.shape)
    train_filename2 = 'compressed_data/' + subject + '_train_poses' + str(st1) + '-' + str(en1) + '.npz'
    np.savez_compressed(train_filename2, train_poses)
    del train_poses
    print('Saved train dataset', train_filename2)

    val_images = load_dataset(path, st1, en1, st2, en2, train=False, read_images=True)
    print('Loaded: ', val_images.shape)
    val_filename1 = 'compressed_data/' + subject + '_val_images' + str(st2) + '-' + str(en2) + '.npz'
    np.savez_compressed(val_filename1, val_images)
    del val_images
    print('Saved validate dataset', val_filename1)

    val_poses = load_dataset(path, st1, en1, st2, en2, train=False, read_images=False)
    print('Loaded: ', val_poses.shape)
    val_filename2 = 'compressed_data/' + subject + '_val_poses' + str(st2) + '-' + str(en2) + '.npz'
    np.savez_compressed(val_filename2, val_poses)
    del val_poses
    print('Saved validate dataset', val_filename2)
    return train_filename1, train_filename2, val_filename1, val_filename2
