import numpy as np
import os

import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline

import cv2
import imageio

from datetime import datetime
from utils import visualize
from utils import results
from object_detection.utils import label_map_util

from evaluate.coco.coco import COCO

MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_PATH = os.path.join('models', MODEL_NAME)
SAVED_MODEL_PATH = os.path.join(MODEL_PATH, 'saved_model')

detection_graph = tf.Graph()
with tf.Session(graph=detection_graph) as sess:
    print('Loading model...')
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], SAVED_MODEL_PATH)
    print('Model loaded')


PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

EVAL_PATH = 'evaluate\coco'
ANN_PATH = os.path.join(EVAL_PATH, 'annotations')
IMG_PATH = os.path.join(EVAL_PATH, 'images')
RES_PATH = os.path.join(EVAL_PATH, 'results')
ann_file_name = 'instances_val2017.json'
ann_file = os.path.join(ANN_PATH, ann_file_name)

# initialize COCO api
coco = COCO(ann_file)

PATH_TO_TEST_IMAGES_DIR = os.path.join(IMG_PATH, 'val2017')


with detection_graph.as_default():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    with tf.Session(graph=detection_graph, config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')
        detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')

        tensor_dict = {'num_detections': num_detections,
                       'detection_boxes': detection_boxes,
                       'detection_scores': detection_scores,
                       'detection_classes': detection_classes}

        total_results = []
        t0 = datetime.now()
        for img in os.listdir(PATH_TO_TEST_IMAGES_DIR)[:1]:
            tf0 = datetime.now()
            image_id = coco.imgToId[img]
            image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, img)

            image = Image.open(image_path)
            image_np = np.array(image)

            if len(image_np.shape) == 2:
                image_np = np.stack((image_np, image_np, image_np), axis=2)

            # run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image_np, 0)},
                                   options=run_options,
                                   run_metadata=run_metadata)

            # all outputs are float32 numpy arrays, so convert types as appropriate
            detections = output_dict['num_detections'] = int(output_dict['num_detections'][0])
            classes = output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            boxes = output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            scores = output_dict['detection_scores'] = output_dict['detection_scores'][0]

        #     for r in results.get_results_by_score_threshold(
        #             image_np,
        #             image_id,
        #             classes,
        #             boxes,
        #             scores,
        #             min_score_thresh=0.3):
        #         total_results.append(r)
        #
        # json_file_name = 'faster_rcnn_inception_v2.json'
        #
        # results.create_json_results_file(
        #     total_results,
        #     os.path.join(RES_PATH, json_file_name))

            # print(i)
            # Create the Timeline object, and write it to a json
            # if i == 100:
            # tl = timeline.Timeline(run_metadata.step_stats)
            # trace = tl.generate_chrome_trace_format()
            # with open(os.path.join('timeline', 'timeline_%d.json' % i), 'w') as f:
            #     f.write(trace)


            image_np = visualize.visualize_boxes_and_labels(
                image_np,
                boxes,
                classes=classes,
                scores=scores,
                category_index=category_index,
                line_width=3,
                label_size=13,
                min_score_thresh=0.6,
                skip_labels_and_scores=False)

            plt.imshow(image_np)
            plt.show()
