import numpy as np
import os

import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image

import datetime
import copy

from tensorflow.core.framework import graph_pb2
from object_detection.utils import label_map_util
from utils import visualize
from utils import results

from evaluate.coco.coco import COCO

def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]


input_graph = tf.Graph()
with tf.Session(graph=input_graph):
    score = tf.placeholder(tf.float32, shape=(None, 1917, 90), name="Postprocessor/convert_scores")
    expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
    for node in input_graph.as_graph_def().node:
        if node.name == "Postprocessor/convert_scores":
            score_def = node
        if node.name == "Postprocessor/ExpandDims_1":
            expand_def = node


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('./models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        dest_nodes = ['Postprocessor/convert_scores', 'Postprocessor/ExpandDims_1']

        edges = {}
        name_to_node_map = {}
        node_seq = {}
        seq = 0

        for node in od_graph_def.node:
            n = _node_name(node.name)
            name_to_node_map[n] = node
            edges[n] = [_node_name(x) for x in node.input]
            node_seq[n] = seq
            seq += 1

        for d in dest_nodes:
            assert d in name_to_node_map, "%s is not in graph" % d

        nodes_to_keep = set()
        next_to_visit = dest_nodes[:]

        while next_to_visit:
            n = next_to_visit[0]
            del next_to_visit[0]
            if n in nodes_to_keep:
                continue
            nodes_to_keep.add(n)
            next_to_visit += edges[n]

        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

        nodes_to_remove = set()
        for n in node_seq:
            if n in nodes_to_keep_list: continue
            nodes_to_remove.add(n)
        nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

        keep = graph_pb2.GraphDef()
        for n in nodes_to_keep_list:
            keep.node.extend([copy.deepcopy(name_to_node_map[n])])

        remove = graph_pb2.GraphDef()
        remove.node.extend([score_def])
        remove.node.extend([expand_def])
        for n in nodes_to_remove_list:
            remove.node.extend([copy.deepcopy(name_to_node_map[n])])

        with tf.device('/gpu:0'):
            tf.import_graph_def(keep, name='')
        with tf.device('/cpu:0'):
            tf.import_graph_def(remove, name='')


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# writer = tf.summary.FileWriter(logdir=os.path.join('tb_graph', MODEL_NAME), graph=detection_graph)
# writer.close()

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
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, img) for img in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

# Size, in inches, of the output images.
# IMAGE_SIZE = (12, 8)


with detection_graph.as_default():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    with tf.Session(graph=detection_graph, config=config) as sess:
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
        expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
        score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
        expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')

        num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')
        detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')

        tensor_dict = {'num_detections': num_detections,
                       'detection_boxes': detection_boxes,
                       'detection_scores': detection_scores,
                       'detection_classes': detection_classes}

        total_results = []
        t0 = datetime.datetime.now()
        # for image_path in TEST_IMAGE_PATHS:
        for img in os.listdir(PATH_TO_TEST_IMAGES_DIR)[:3]:
        # img = '000000007888.jpg'
            image_id = coco.imgToId[img]
            image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, img)
            # image_path = TEST_IMAGE_PATHS[48]

            image = Image.open(image_path)
            image_np = np.array(image)

            if len(image_np.shape) == 2:
                image_np = np.stack((image_np, image_np, image_np), axis=2)

            # run inference
            (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: np.expand_dims(image_np, 0)})
            output_dict = sess.run(tensor_dict, feed_dict={score_in: score, expand_in: expand})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            # output_dict['num_detections'] = int(output_dict['num_detections'][0])
            classes = output_dict['detection_classes'][0].astype(np.uint8)
            boxes = output_dict['detection_boxes'][0]
            scores = output_dict['detection_scores'][0]

        #     for r in results.get_results_by_score_threshold(
        #             image_np,
        #             image_id,
        #             classes,
        #             boxes,
        #             scores,
        #             min_score_thresh=0.3):
        #         total_results.append(r)
        #
        # json_file_name = 'val2017_ssd_mobilenet_0.3.json'
        #
        # results.create_json_results_file(
        #     total_results,
        #     os.path.join(RES_PATH, json_file_name))

            # print(img)

        # print((datetime.datetime.now() - t0).total_seconds())
            image_np = visualize.visualize_boxes_and_labels(
                image_np,
                boxes,
                classes=classes,
                scores=scores,
                category_index=category_index,
                line_width=3,
                label_size=13,
                min_score_thresh=0.5,
                skip_labels_and_scores=False)

            # plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
