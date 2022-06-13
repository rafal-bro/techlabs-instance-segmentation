"""
This module contains helper functions for Mask R-CNN initialization, training, and detection.

Written by Rafal Broda
Last updated 04.04.2022
"""

import sys
import os
import json
import random
import shutil

import numpy as np
from sklearn import metrics
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon


# Import Mask RCNN
sys.path.append('../') 
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log

""" NOTEBOOK PREFERENCES """

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images.
    From https://pysource.com/.
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


""" DATASET """

class CustomDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
        From https://pysource.com/.
    """

    def load_custom(self, annotation_json, images_dir, dataset_type="train"):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """

        # Load json from file
        print("Annotation json path: ", annotation_json)
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()


        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']

            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}

        # Split the dataset, if train, get 80%, if val, get 20%, if test do not split
        len_images = len(coco_json['images'])
        if dataset_type == "train":
            img_range = [int(len_images * 0.2), len_images]
        elif dataset_type == "val":
            img_range = [0, int(len_images * 0.2) ]
        elif dataset_type == "test":
            img_range = [0, len_images]

        for i in range(img_range[0], img_range[1]):
            image = coco_json['images'][i]
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        #print("Class_ids, ", class_ids)
        return mask, class_ids

    def count_classes(self):
        class_ids = set()
        for image_id in self.image_ids:
            image_info = self.image_info[image_id]
            annotations = image_info['annotations']

            for annotation in annotations:
                class_id = annotation['category_id']
                class_ids.add(class_id)

        class_number = len(class_ids)
        return class_number


def extract_images(my_zip, output_dir):
    """
    Extract dataset from zip.
    From https://pysource.com/.
    Args:
        my_zip (str): Path zip file
        output_dir (str): Desired output path
    Returns:
        None
    """

    # Make directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(my_zip) as zip_file:
        count = 0
        for member in zip_file.namelist():
            filename = os.path.basename(member)
            # skip directories
            if not filename:
                continue
            count += 1
            # copy file (taken from zipfile's extract)
            source = zip_file.open(member)
            target = open(os.path.join(output_dir, filename), "wb")
            with source, target:
                shutil.copyfileobj(source, target)
        print("Extracted: {} images".format(count))


def load_image_dataset(annotation_path, dataset_path, dataset_type):
    """
    Load and initialize dataset for model training. Perform train and val split.
    From https://pysource.com/ and modified.
    Args:
        annotation_path (str): Path to coco file
        dataset_path (str): Path to filder containing images
        dataset_type (str): Either train, val or test
                            when choosing train 80% will be chosen,
                            when choosing val the remaining 20% will be chosen,
                            when choosing test 100% will be returned since train-test split was performed before
    Returns:
        dataset_train (object): Training dataset ready for Mask R-CNN training
    """
    dataset_train = CustomDataset()
    dataset_train.load_custom(annotation_path, dataset_path, dataset_type)
    dataset_train.prepare()
    return dataset_train


def display_image_samples(dataset_train):
    """
    Show some images from previously loaded dataset.
    From https://pysource.com/.
    Args:
        dataset_train (object): Training dataset ready for Mask R-CNN training
    Returns:
        None
    """
    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 4)

    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


def count_instances(coco_path):
    """
    Count and print distribution of instances in coco file.
    Args:
        coco_path (str): Path to coco file
    Returns:
        None
    """

    file = open(coco_path)
    d = json.loads(file.read())

    instances = {'aardvark_hole':0,'bush':0,'animal':0,'road':0,'dead_tree':0,'tree':0}

    for i in range(len(d['annotations'])):
        annot = d['annotations'][i]
        for j in range(len(d['categories'])):
            if (annot['category_id'] == d['categories'][j]['id']):
                obj = d['categories'][j]['name']
                instances[obj] = instances[obj] + 1

    print('instances',instances)


""" TRAINING MODEL """

def load_training_model(train_config, MODEL_DIR, ROOT_DIR, init_with="coco"):
    """
    Load and initialize model for training.
    From https://pysource.com/ and modified.
    Args:
        train_config (object): Training configuration
        MODEL_DIR (str): Path to directory containing trained models and logs
        ROOT_DIR (str): Root to Mask R-CNN directory
        init_with (str): Model initialization - choose between coco, imagenet, and last
    Returns:
        model (object): Model with loaded weights
    """

    model = modellib.MaskRCNN(mode="training", config=train_config,
                              model_dir=MODEL_DIR)

    # Which weights to start with? imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
        print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    return model

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
def train_head(model, dataset_train, dataset_val, config, epochs=5):
    """From https://pysource.com/"""
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs,
            layers='heads')


def train_all_layers(model, dataset_train, dataset_val, config, epochs=5):
    """From https://pysource.com/"""
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epochs,
                layers="all")


""" TESTING AND DETECTION """

def load_inference_model(inference_config, MODEL_DIR):
    """
    Load and initialize model in inference mode for testing and detection.
    From https://pysource.com/.
    Args:
        inference_config (object): Inference configuration
        MODEL_DIR (str): Path to directory containing trained models and logs
    Returns:
        model (object): Model with loaded weights
    """
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    #model_path = os.path.join(ROOT_DIR, ".h5")
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


def test_random_image(test_model, dataset_test, inference_config):
    """
    Perform test on random image.
    From https://pysource.com/.
    Args:
        test_model (object): Trained model initialized in inference mode
        dataset_val (object): Validation dataset
        inference_config (object): Inference configuration
    Returns:
        None
    """
    image_id = random.choice(dataset_test.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # Model result
    print("Trained model result")
    results = test_model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_test.class_names, r['scores'], ax=get_ax(), show_bbox=False, title='detection')

    print("Annotation")
    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_test.class_names, figsize=(8, 8), title='ground truth')


def compute_ar(pred_boxes, gt_boxes, list_iou_thresholds):
    """
    Compute average recall.
    Args:
        pred_boxes (array): Boxes predicted by model
        gt_boxes (array): Ground-truth boxes
        list_iou_thresholds (float): Threshold between 0 and 1 for
    Returns:
        AUC (float): Area under curve
    """
    AR = []
    for iou_threshold in list_iou_thresholds:

        try:
            recall, _ = utils.compute_recall(pred_boxes, gt_boxes, iou=iou_threshold)

            AR.append(recall)

        except:
          AR.append(0.0)
          pass

    AUC = 2 * (metrics.auc(list_iou_thresholds, AR))
    return AUC


def evaluate_model(dataset_test, test_model, inference_config, list_iou_thresholds=None):
    """
    Compute mean average precision, mean average recall and f1 score.
    Args:
        dataset_test (object): Test dataset
        test_model (object): Trained model initialized in inference mode
        inference_config (object): Inference configuration
        list_iou_thresholds (float): Threshold between 0 and 1 for
    Returns:
        mAP (float): Mean average precisions
        mAR (float): Mean average recalls
        f1_score (float): F1 score
    """
    if list_iou_thresholds is None: list_iou_thresholds = np.arange(0.5, 1.01, 0.1)

    APs = []
    ARs = []
    for image_id in dataset_test.image_ids:

        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, inference_config, image_id)

        scaled_image = modellib.mold_image(image, inference_config)

        sample = np.expand_dims(scaled_image, 0)

        yhat = test_model.detect(sample, verbose=0)

        r = yhat[0]

        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.5)

        AR = compute_ar(r['rois'], gt_bbox, list_iou_thresholds)
        ARs.append(AR)
        APs.append(AP)

    mAP = np.mean(APs)
    mAR = np.mean(ARs)
    f1_score = 2 * ((mAP * mAR) / (mAP + mAR))


    return mAP, mAR, f1_score
    
def display_instance_and_img(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(9, 18), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    
    plt.figure(figsize=figsize)

    
    # equivalent but more general
    ax = plt.subplot(1, 2, 1)
    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_ylim(height + 10, -10)
    ax2.set_xlim(-10, width + 10)
    ax2.axis('off')
    ax2.set_title('original')
    plt.imshow(image)
    plt.show()
