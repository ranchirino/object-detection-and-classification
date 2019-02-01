import numpy as np
import os

from utils import visualize

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import cv2
import imageio

from datetime import datetime
from object_detection.utils import label_map_util

MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_PATH = os.path.join('models', MODEL_NAME)
SAVED_MODEL_PATH = os.path.join(MODEL_PATH, 'saved_model')

detection_graph = tf.Graph()
with tf.Session(graph=detection_graph) as sess:
    print('Loading model...')
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], SAVED_MODEL_PATH)
    print('Model loaded')

    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "DESKTOP-66LMRU5:7000")

    # writer = tf.summary.FileWriter(logdir=os.path.join('tb_graph', MODEL_NAME), graph=detection_graph)
    # writer.close()

    # ops = [op for op in detection_graph.get_operations()]
    #
    # sess.run(tf.global_variables_initializer())
    #
    # for v in tf.global_variables():
    #     print(v.name)

    # tr = [v for v in tf.trainable_variables()]
    # for v in tf.trainable_variables():
    #     print(v.name)

    # log_dir = "/log_dir"
    # out_file = "train.pbtxt"
    # tf.train.write_graph(detection_graph, logdir=log_dir, name=out_file, as_text=True)


from tensorflow.python import pywrap_tensorflow

model_file = os.path.join(MODEL_PATH, 'model.ckpt')
reader = pywrap_tensorflow.NewCheckpointReader(model_file)
var_to_shape_map = reader.get_variable_to_shape_map()

# tensor_names = [key for key in var_to_shape_map if 'weights' in str(key)]

tensor = reader.get_tensor('FirstStageFeatureExtractor/InceptionV2/Conv2d_1a_7x7/pointwise_weights')

TENSORS_PATH = os.path.join('data', 'tensors')

np.save(os.path.join(TENSORS_PATH, 'weights'), tensor)

# conv_tensor_names = [op.name for op in detection_graph.get_operations() if op.type=='Conv2D']

# tensor = reader.get_tensor(conv_tensor_names[0])

# from tensorflow.python.tools import inspect_checkpoint as chkp
# chkp.print_tensors_in_checkpoint_file(os.path.join(MODEL_PATH, 'model.ckpt'), tensor_name='', all_tensors=True)


#%%

# writer = tf.summary.FileWriter(logdir=os.path.join('tb_graph', MODEL_NAME), graph=detection_graph)
# writer.close()


# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#
# #%%
# PATH_TO_TEST_VIDEOS_DIR = 'test_videos'
# video_name = 'toronto_1.mp4'
# VIDEO = os.path.join(PATH_TO_TEST_VIDEOS_DIR, video_name)
# cap = cv2.VideoCapture(VIDEO)
#
# # reader = imageio.get_reader(VIDEO)
# # fps_video = reader.get_meta_data()['fps']
# # writer = imageio.get_writer('video_faster_rcnn_incepv2_1.mp4', fps=fps_video)
#
#
# with detection_graph.as_default():
#     config = tf.ConfigProto()
#     config.allow_soft_placement = True
#     with tf.Session(graph=detection_graph, config=config) as sess:
#         image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
#
#         # score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
#         # expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
#         # score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
#         # expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
#
#         num_detections = tf.get_default_graph().get_tensor_by_name('num_detections:0')
#         detection_boxes = tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
#         detection_scores = tf.get_default_graph().get_tensor_by_name('detection_scores:0')
#         detection_classes = tf.get_default_graph().get_tensor_by_name('detection_classes:0')
#
#         tensor_dict = {'num_detections': num_detections,
#                        'detection_boxes': detection_boxes,
#                        'detection_scores': detection_scores,
#                        'detection_classes': detection_classes}
#
#         t0 = datetime.now()
#         n_frame = 0
#         acc_inf_speed = 0 # average inference speed
#         acc_fps = 0
#         # total_time = 0 # total time without counting the 1st frame
#         # for i, frame in enumerate(reader):
#         while(cap.isOpened()):
#             ret, frame = cap.read()
#             tf0 = datetime.now()
#             n_frame += 1
#             frame_np = np.array(frame)
#
#
#             t1 = datetime.now()
#             # run inference
#             # (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: np.expand_dims(frame_np, 0)})
#
#             # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#             # run_metadata = tf.RunMetadata()
#
#             output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(frame_np, 0)})
#             inf_speed = (datetime.now() - t1).total_seconds() * 1000 # inference speed per frame
#             # print('Inference speed: %.2f ms' % (inf_speed))
#
#             # all outputs are float32 numpy arrays, so convert types as appropriate
#             output_dict['num_detections'] = int(output_dict['num_detections'][0])
#             output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
#             output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
#             output_dict['detection_scores'] = output_dict['detection_scores'][0]
#
#             tv0 = datetime.now()
#             # frame_np = visualize.visualize_boxes_and_labels(
#             #     frame_np,
#             #     output_dict['detection_boxes'],
#             #     classes=output_dict['detection_classes'],
#             #     scores=output_dict['detection_scores'],
#             #     category_index=category_index,
#             #     line_width=3,
#             #     skip_labels_and_scores=False)
#             visual_time = (datetime.now() - tv0).total_seconds() * 1000 # visualization time
#
#             if n_frame != 1:
#                 time_per_frame = (datetime.now() - tf0).total_seconds() * 1000 # processing time per frame
#                 fps = 1000 / time_per_frame
#                 acc_fps = acc_fps + fps
#                 avg_fps = acc_fps / (n_frame - 1)
#                 acc_inf_speed += inf_speed
#                 frame_np = visualize.draw_text(frame_np, ['Inference speed: {:.2f} ms, Time per frame: {:.2f} ms, Average fps: {:.2f}'.format(inf_speed, time_per_frame, avg_fps)], [(20,20)])
#
#             # writer.append_data(frame_np)
#             print('Inference speed: %.2f ms, Visualization time: %.2f ms' % (inf_speed, visual_time))
#
#             cv2.imshow('Object Detection', frame_np)
#             if cv2.waitKey(25) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break
#
#         avg_inf_speed = acc_inf_speed / (n_frame - 1)
#         # fps = n_frame / (datetime.now() - t0).total_seconds()
#         print('Average inference speed: %.2f ms' % (avg_inf_speed))
#         # print('Fps: %.2f' % (fps))
#         # writer.close()
#
#         # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#         # chrome_trace = fetched_timeline.generate_chrome_trace_format()
#         # with open('Experiment_1.json', 'w') as f:
#         #     f.write(chrome_trace)

