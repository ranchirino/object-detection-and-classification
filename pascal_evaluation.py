import os
import numpy as np
from evaluate.pascal import pascal
from object_detection.utils import label_map_util

IMG_PATH = 'C:\RANGEL\GitHub\evaluation_images'
IMG_PATH_PASCAL = os.path.join(IMG_PATH, 'pascal')
ANN_PATH_PASCAL = os.path.join(IMG_PATH_PASCAL, 'annotations')

EVAL_PATH = 'evaluate'
EVAL_PATH_PASCAL = os.path.join(EVAL_PATH, 'pascal')

IMG_SETS_PASCAL = os.path.join(EVAL_PATH_PASCAL, 'image_sets\main')
RES_PATH_PASCAL = os.path.join(EVAL_PATH_PASCAL, 'results')

val_set = open(os.path.join(IMG_SETS_PASCAL, 'val.txt'), "r")
image_ids = [id[:-1] for id in val_set]
val_set.close()

PATH_TO_LABELS_PASCAL = os.path.join('data', 'pascal_label_map_v2.pbtxt')
categories_pascal = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_PASCAL, use_display_name=True)
categories = [categories_pascal[item]['name'] for item in categories_pascal]

#%% get ground truth annotations
gt_annot = pascal.get_det_annot(ANN_PATH_PASCAL, image_ids)

# get detections results
det_result = pascal.get_det_results_by_category_files(RES_PATH_PASCAL, categories)

# obtain the true positives and the false positives by category,
# the precision, the recall, and the average precision
avg_prec = []
for categ in categories:
    gt = [gt for gt in gt_annot if gt['category'] == categ]
    det = [det for det in det_result if det['category'] == categ]

    tp, fp = pascal.obtain_tp_and_fp_by_category(gt, det)
    precision, recall = pascal.compute_precision_and_recall(tp, fp, len(gt))
    ap = pascal.average_precision(precision, recall)
    # pascal.plot_precision_recall_curve(precision, recall, categ, ap)
    avg_prec.append({'category': categ,
                     'ap': ap})
    pascal.save_ap_by_category(avg_prec, 'avg_prec', RES_PATH_PASCAL)
    map = pascal.mean_average_precision(avg_prec)