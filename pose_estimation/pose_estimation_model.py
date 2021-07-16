from os.path import join, isfile
import os
import cv2 as cv
import tqdm
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose,heatmap_to_coord
import cv2 as cv
import mxnet as mx
import numpy as np
from mxnet import nd, image


detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

detector.reset_class(["person"], reuse_weights=['person'])

def get_poses_using_model(frames_path, poses_path):
  if not os.path.exists(poses_path):
    os.makedirs(poses_path)

  frames_names = os.listdir(frames_path)
  frames_names = sorted(frames_names)
  frames_count = len(frames_names)
  frames_full_names = [frames_path + '/' + frames_names[i] for i in range(frames_count)]

  for i in range (50, frames_count):
    im_fname = cv.imread(frames_full_names[i])
    img = mx.nd.array(im_fname)
    x, img = data.transforms.presets.ssd.transform_test(img, short=512)
    print('Shape of pre-processed image:', x.shape)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)

    if pose_input is None:
      continue

    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    white_img = np.zeros((im_fname.shape[0], im_fname.shape[1], im_fname.shape[2]))

    bounding_boxs =  bounding_boxs.asnumpy()
    bounding_boxs[bounding_boxs != -1] = -1
    scores[scores != -1] = -1
    white_img[white_img == 0] = 255
    
    ax = utils.viz.plot_keypoints(white_img, pred_coords, confidence,
                                class_IDs, bounding_boxs, scores)
    ax.axis('off')
    plt.savefig(poses_path + '/' + frames_names[i])
    plt.clf()
