"""
Helper functions for data postprocessing for Mask R-CNN model inference.
Used for data capturing and counting of detections.

Written by Mrunmai Phatak and Rafal Broda
Last updated 07.04.2022
"""

import os

import numpy as np
import skimage

def capture_detections(detection_result, img_size):
  """
  Capture and detections from Mask R-CNN inference model.
  Args:
      detection_result (dict): Dictionary resulting from Mask R-CNN inference
      img_size (int):  Image size model was trained with
  Returns:
      List of detection counts
  """
  cnt_tree = 0
  cnt_bush = 0
  cnt_animal = 0
  cnt_deadtree = 0
  cnt_aardvarkhole = 0
  area_tree = 0
  area_bush = 0
  area_road = 0

  for i in range(len(detection_result["rois"])):
    instance = detection_result["rois"][i]
    if detection_result["class_ids"][i] == 1:
      area = (instance[2] - instance[0]) * (instance[3] - instance[1])
      area_bush += area
      cnt_bush += 1
    elif detection_result["class_ids"][i] == 2:
      cnt_aardvarkhole +=1
    elif detection_result["class_ids"][i] == 3:
      cnt_deadtree +=1
    elif detection_result["class_ids"][i] == 4:
      area = (instance[2] - instance[0]) * (instance[3] - instance[1])
      area_tree += area
      cnt_tree += 1
    elif detection_result["class_ids"][i] == 5:
      area = (instance[2] - instance[0]) * (instance[3] - instance[1])
      area_road += area
    elif detection_result["class_ids"][i] == 6:
      cnt_animal += 1

    area_vegetation = area_tree + area_bush
    percentage_vegetation = np.round((area_vegetation/(img_size*img_size))*100,1)
    cnt_vegetation = cnt_tree + cnt_bush
    image = detection_result["image"]

  return [image, cnt_tree, cnt_bush, cnt_vegetation, cnt_animal, cnt_deadtree, cnt_aardvarkhole, area_tree, area_bush, area_vegetation, percentage_vegetation, area_road]


def get_detection_dict(model, img_size, inference_path, inference_list=None):
    """
    Perform detection on all images given and capture results in a dictionary.
    Args:
        model (object): Mask R-CNN inference model
        img_size (int):  Image size model was trained with
        inference_path (str): Path to directory containing raw image data for inference
        inference_list (list, default:None): List containing image filenames to perform inference on - optional, if not provided all images from inference_path will be used
    Returns:
        detection_dict (dict): Dictionary contianing detections, can be easily transformed into pandas DataFrame for data analysis
    """
    if inference_list == None:
        file_names = next(os.walk(inference_path))[2]
    else:
        fine_names = inference_list

    detection_dict = {}

    detection_dict['image'] = []
    detection_dict['cnt_tree'] = []
    detection_dict['cnt_bush'] = []
    detection_dict['cnt_vegetation'] = []
    detection_dict['cnt_animal'] = []
    detection_dict['cnt_deadtree'] = []
    detection_dict['cnt_aardvarkhole'] = []
    detection_dict['area_tree'] = []
    detection_dict['area_bush'] = []
    detection_dict['area_vegetation'] = []
    detection_dict['percentage_vegetation'] = []
    detection_dict['area_road'] = []

    # perform detection on all images and save results as pkl file
    for file in file_names:
        try:
          image = skimage.io.imread(os.path.join(inference_path, file))
          results = model.detect([image], verbose=0)
          results[0]['image'] = file

          # for each image one pickle file will be created
          out = capture_detections(results[0], img_size)

          detection_dict['image'].append(out[0])
          detection_dict['cnt_tree'].append(out[1])
          detection_dict['cnt_bush'].append(out[2])
          detection_dict['cnt_vegetation'].append(out[3])
          detection_dict['cnt_animal'].append(out[4])
          detection_dict['cnt_deadtree'].append(out[5])
          detection_dict['cnt_aardvarkhole'].append(out[6])
          detection_dict['area_tree'].append(out[7])
          detection_dict['area_bush'].append(out[8])
          detection_dict['area_vegetation'].append(out[9])
          detection_dict['percentage_vegetation'].append(out[10])
          detection_dict['area_road'].append(out[11])

        except:
            print(f'image {file} skipped')
    return detection_dict
