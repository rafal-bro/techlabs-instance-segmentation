"""
Helper functions for data preprocessing for Mask R-CNN model training.
Used for data cleaning and preparation of kuzikus dataset in coco-format.

Written by Rafal Broda
Last updated 06.04.2022
"""

import os
import json
import shutil

import funcy
import labelme2coco
from sahi.slicing import slice_coco, slice_image
from sahi.utils.file import load_json, save_json
from sklearn.model_selection import train_test_split

def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'images': images, 'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def create_coco(raw_data_path, img_format, data_path):
    """
    Create coco dataset with json file for Mask R-CNN model training.
    Args:
        raw_data_path (str): Path to folder containing raw images as well as images with annotations
        img_format (str): Format of images (e.g. jpg, png .. )
        data_path (str): Path to desired folder for results
    Returns:
        None
    """

    # create new folder containing clean data for coco dataset
    if not os.path.exists(os.path.join(data_path, os.path.split(data_path)[-1]+'.json')):
        os.mkdir(data_path)

        # find images with corresponding annotations
        img_files = []
        json_files = []
        
        all_files = os.listdir(raw_data_path)

        for file in all_files:
            if file.endswith(img_format):
                img_files.append(os.path.splitext(file)[0])
            elif file.endswith('json'):
                json_files.append(os.path.splitext(file)[0])

        imgs_with_ann = list(set(img_files) & set(json_files))

        for file in imgs_with_ann:
            shutil.copy(os.path.join(raw_data_path, file+f'.{img_format}'), data_path)
            shutil.copy(os.path.join(raw_data_path, file+'.json'), data_path)

        # convert labelme annotations to coco
        labelme2coco.convert(data_path, os.path.join(data_path, os.path.split(data_path)[-1]+'.json'))
    else:
        print(f'Coco dataset already exists at: {data_path}')


def crop_images_coco(data_path, img_size):
    """
    Crop images and according annotations from coco annotaion file for Mask R-CNN model training.
    Args:
        data_path (str): Path to folder containing coco dataset
        img_size (int): Desired image size
    Returns:
        None
    """

    if not os.path.exists(data_path+f'_{img_size}'):
        os.mkdir(data_path+f'_{img_size}')

        coco_dict, _ = slice_coco(coco_annotation_file_path=os.path.join(data_path, os.path.split(data_path)[-1]+'.json'),
                                  ignore_negative_samples=True,
                                  image_dir = '',
                                  output_coco_annotation_file_name=False,
                                  output_dir=data_path+f'_{img_size}',
                                  slice_height=img_size,
                                  slice_width=img_size,
                                  overlap_height_ratio=0,
                                  overlap_width_ratio=0,
                                  min_area_ratio=0,
                                  verbose=False
                                  )

        images = coco_dict['images']
        annotations = coco_dict['annotations']
        categories = coco_dict['categories']
        
        idx_list = []
        
        for idx, i in enumerate(annotations):
            if i['bbox'] == []:
                idx_list.append(idx)
        
        for i in reversed(idx_list):
            del(annotations[i])
                
        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        save_coco(os.path.join(data_path+f'_{img_size}', os.path.split(data_path)[-1]+f'_{img_size}'+'.json'), images, filter_annotations(annotations, images), categories)

        print(f'Saved {len(images)} entries in {data_path}_{img_size}')

    else:
        print(f'Cropped dataset already exists at: {data_path}_{img_size}')



def crop_images_inference(data_path, raw_data_path, img_size, img_format):
    """
    Crop images for the inference model. Beforehand images used for model development (train, val, test) will be sorted out.
    Args:
        data_path (str): Path to folder containing coco dataset
        images_path (str): Path to all images
        inference_path (str): Path to folder where results should be saved
    Returns:
        None
    """
    data_path = data_path+f'_{img_size}'
    
    inference_path = data_path+'_inference'

    if not os.path.exists(inference_path):
       os.mkdir(inference_path)

       # get image filenames used for model development
       img_list = os.listdir(data_path)

       # get filenames of all images
       all_imgs = os.listdir(raw_data_path)
       
       model_imgs = []

       for i in img_list:
           if i.endswith('jpg') or i.endswith('png'):
               model_imgs.append(os.path.splitext(i)[0]+f'.{img_format}')

       # filter out duplicates
       model_imgs = list(set(model_imgs))

       # filter out images used for training and testing
       filtered_imgs = [x for x in all_imgs if x not in model_imgs]

       inference_imgs = []

       for file in filtered_imgs:
           if file.endswith(img_format):
               inference_imgs.append(file)

       # create inference dataset by cropping remaining images and copying them into the inference folder

       for file in inference_imgs:
           #try:
               sliced_image_result = slice_image(
                                image=os.path.join(raw_data_path, file),
                                output_file_name=file.split('.')[0],
                                output_dir=inference_path,
                                slice_height=img_size,
                                slice_width=img_size,
                                overlap_height_ratio=0,
                                overlap_width_ratio=0,
                                min_area_ratio=0,
                                verbose=False
                                )
           #except:
           #     pass

       print(f'{len(os.listdir(inference_path))} images saved in {inference_path}')
    else:
      print(f'Inference dataset already exists at: {inference_path}')


def split_coco(coco_path, split=0.8):
    """
    Perform train test split.
    Args:
        coco_path (str): Path to coco file
        split (float, default:0.8): Size of train dataset, between 0 and 1
    """
    file = open(coco_path)
    coco = json.loads(file.read())
    images = coco['images']
    annotations = coco['annotations']
    categories = coco['categories']

    number_of_images = len(images)

    x, y = train_test_split(images, train_size=split, shuffle=False)

    save_coco(os.path.join(os.path.split(coco_path)[0], 'train.json'), x, filter_annotations(annotations, x), categories)
    save_coco(os.path.join(os.path.split(coco_path)[0], 'test.json'), y, filter_annotations(annotations, y), categories)
