import os
import numpy as np
import matplotlib.pyplot as plt
from evaluate.pascal import pascal
from object_detection.utils import label_map_util
from utils import visualize, results
from PIL import Image

IMG_PATH = 'C:\RANGEL\GitHub\evaluation_images'
IMG_PATH_PASCAL = os.path.join(IMG_PATH, 'pascal')
ANN_PATH_PASCAL = os.path.join(IMG_PATH_PASCAL, 'annotations')

EVAL_PATH = 'evaluate'
EVAL_PATH_PASCAL = os.path.join(EVAL_PATH, 'pascal')

IMG_SETS_PASCAL = os.path.join(EVAL_PATH_PASCAL, 'image_sets\main')
RES_PATH_PASCAL = os.path.join(EVAL_PATH_PASCAL, 'results')
PATH_TO_TEST_IMAGES = os.path.join(IMG_PATH_PASCAL, 'voc2012')

val_set = open(os.path.join(IMG_SETS_PASCAL, 'val.txt'), "r")
image_ids = [id[:-1] for id in val_set]
val_set.close()

PATH_TO_LABELS_PASCAL = os.path.join('data', 'pascal_label_map_v2.pbtxt')
categories_pascal = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_PASCAL, use_display_name=True)
categories = [categories_pascal[item]['name'] for item in categories_pascal]

#%% get ground truth annotations
gt_annot = pascal.get_gt_annot(ANN_PATH_PASCAL, image_ids, difficult=False)
# get detections results
det_result = pascal.get_det_results_by_category_files(RES_PATH_PASCAL, categories)

# map_by_score = []
# map = 0
# for score in np.arange(0.5, 0.96, 0.01):
#     print('Score: %.2f' % score)
#     det_by_score = [d for d in det_result if d['score'] >= score]

det_by_score = [d for d in det_result if d['score'] >= 0.76]
# obtain the true positives and the false positives by category,
# the precision, the recall, and the average precision
avg_prec = []
det_evaluated = []
for categ in categories:
    gt = [gt for gt in gt_annot if gt['category'] == categ]
    det = [det for det in det_by_score if det['category'] == categ]

    det_tp, tp, fp = pascal.obtain_tp_and_fp_by_category(gt, det, image_ids)
    for d in det_tp:
        det_evaluated.append(d)
    precision, recall = pascal.compute_precision_and_recall(tp, fp, len(gt))
    ap = pascal.interpolated_average_precision(precision, recall)
    # pascal.plot_precision_recall_curve(precision, recall, categ, ap)
    avg_prec.append({'category': categ,
                        'ap': ap})
    # pascal.save_ap_by_category(avg_prec, 'avg_prec_%.2f' % score, RES_PATH_PASCAL)
    map = pascal.mean_average_precision(avg_prec)

# map_by_score.append({'score': score, 'map': map})


# # visualize an image with detections and ground truth
# image_id = image_ids[np.random.randint(0, len(image_ids))]
# gt = [gt for gt in gt_annot if gt['image_id'] == image_id]
# det = [det for det in det_evaluated if det['image_id'] == image_id]
# image_path = os.path.join(PATH_TO_TEST_IMAGES, image_id + '.jpg')
# image = Image.open(image_path)
# image_np = np.array(image)
# image_np = visualize.visualize_det_and_gt(image_np, det, gt, line_width=3, label_size=13, show_tp_and_fp=True)
# plt.imshow(image_np)
# plt.show()


# plot map by score
map_by_score = results.read_data_from_xml_file('map_by_score', RES_PATH_PASCAL)
map_to_plot = [float(m['map']) for m in map_by_score]
plt.plot(np.arange(0.5, 0.96, 0.01), map_to_plot, 'bo')
plt.ylabel('mAP')
plt.xlabel('Confidence score')
plt.title('mAP by confidence score')
plt.grid(True)
plt.show()