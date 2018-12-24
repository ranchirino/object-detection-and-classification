import numpy as np
import os
import sys

import tensorflow as tf

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import imageio
from skimage import transform

import time
from datetime import datetime
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

# def load_image_into_numpy_array(image):
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)

def draw_image(np_img, text, pos):
    image = Image.fromarray(np_img)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arial.ttf', 24)
    color = 'rgb(0, 0, 0)'
    draw.text(pos, text, fill=color, font=font)
    return  np.array(image)


#%%
PATH_TO_TEST_VIDEOS_DIR = 'test_videos'
video_name = 'new_york_city_1.mp4'
VIDEO = os.path.join(PATH_TO_TEST_VIDEOS_DIR, video_name)
cap = cv2.VideoCapture(VIDEO)

# reader = imageio.get_reader(VIDEO)
# fps_video = reader.get_meta_data()['fps']
# writer = imageio.get_writer('video_detection.mp4', fps=fps_video)

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

        t0 = datetime.now()
        n_frame = 0
        tot_inf_speed = 0 # average inference speed
        # for i, frame in enumerate(reader):
        while(cap.isOpened()):
            ret, frame = cap.read()
            tf0 = datetime.now()
            n_frame += 1
            frame_np = np.array(frame)

            # resize image
            # frame_rs = transform.resize(frame_np, (337, 600))

            t1 = datetime.now()
            # run inference
            (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: np.expand_dims(frame_np, 0)})
            output_dict = sess.run(tensor_dict, feed_dict={score_in: score, expand_in: expand})
            inf_speed = (datetime.now() - t1).total_seconds() * 1000 # inference speed per frame
            # print('Inference speed: %.2f ms' % (inf_speed))

            if n_frame != 1:
                tot_inf_speed += inf_speed

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            tv0 = datetime.now()
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     frame_np,
            #     output_dict['detection_boxes'],
            #     output_dict['detection_classes'],
            #     output_dict['detection_scores'],
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=4)
            visual_time = (datetime.now() - tv0).total_seconds() * 1000 # visualization time

            time_per_frame = (datetime.now() - tf0).total_seconds() * 1000 # processing time per frame
            # image = draw_image(frame_np, 'Inference speed: {:.2f} ms'.format(inf_speed), (20,20))
            # image = draw_image(image, 'Processing time per frame: {:.2f} ms'.format(time_per_frame), (310,20))
            # writer.append_data(image)
            print('Inference speed: %.2f ms, Visualization time: %.2f ms' % (inf_speed, visual_time))

            cv2.imshow('Object Detection', frame_np)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        avg_inf_speed = tot_inf_speed / n_frame
        fps = n_frame / (datetime.now() - t0).total_seconds()
        print('Average inference speed: %.2f ms' % (avg_inf_speed))
        print('Fps: %.2f' % (fps))
        # writer.close()


