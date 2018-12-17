import numpy as np
import os
import sys

import tensorflow as tf

# import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

import time
import datetime
import copy

from tensorflow.core.framework import graph_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


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
        # print(node.name)
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

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        # config.log_device_placement = True
        with tf.Session(graph=graph, config=config) as sess:
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')

            detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
            detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
            detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
            num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')

            tensor_dict = {'num_detections': num_detections,
                           'detection_boxes': detection_boxes,
                           'detection_scores': detection_scores,
                           'detection_classes': detection_classes}

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            if 'detection_masks:0' in all_tensor_names:
                tensor_dict['detection_masks'] = tf.get_default_graph().get_tensor_by_name('detection_masks:0')

            (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict = sess.run(tensor_dict, feed_dict={score_in: score, expand_in: expand})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict



for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = load_image_into_numpy_array(image)
    # image_np_expanded = np.expand_dims(image_np, axis=0)

    t1 = datetime.datetime.now()
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    t2 = datetime.datetime.now()
    print(t2 - t1)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=5)

    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()



# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#
#         score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
#         expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
#         score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
#         expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
#
#         detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#         detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#         detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#         num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # i = 0
        # for _ in range(10):
        #     image_path = TEST_IMAGE_PATHS[1]
        #     i += 1
        #     image = Image.open(image_path)
        #     image_np = load_image_into_numpy_array(image)
        #     image_np_expanded = np.expand_dims(image_np, axis=0)
        #
        #     start_time = time.time()
        #     (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: image_np_expanded})
        #
        #     (boxes, scores, classes, num) = sess.run(
        #         [detection_boxes, detection_scores, detection_classes, num_detections],
        #         feed_dict={score_in: score, expand_in: expand})
        #     print('Iteration %d: %.3f sec' % (i, time.time() - start_time))
        #
        #     vis_util.visualize_boxes_and_labels_on_image_array(
        #         image_np,
        #         np.squeeze(boxes),
        #         np.squeeze(classes).astype(np.int32),
        #         np.squeeze(scores),
        #         category_index,
        #         use_normalized_coordinates=True,
        #         line_thickness=5)


        # for image_path in TEST_IMAGE_PATHS:
        #     image = Image.open(image_path)
        #     image_np = load_image_into_numpy_array(image)
        #     image_np_expanded = np.expand_dims(image_np, axis=0)

        #     (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: image_np_expanded})
        #     (boxes, scores, classes, num) = sess.run(
        #         [detection_boxes, detection_scores, detection_classes, num_detections],
        #         feed_dict={score_in: score, expand_in: expand})

        #     vis_util.visualize_boxes_and_labels_on_image_array(
        #         image_np,
        #         np.squeeze(boxes),
        #         np.squeeze(classes).astype(np.int32),
        #         np.squeeze(scores),
        #         category_index,
        #         use_normalized_coordinates=True,
        #         line_thickness=5)

        #     plt.figure(figsize=IMAGE_SIZE)
        #     plt.imshow(image_np)
        #     plt.show()
