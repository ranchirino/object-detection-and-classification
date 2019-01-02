import numpy as np
import os
from evaluate.coco.coco import COCO
from evaluate.coco.cocoeval import COCOeval
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

EVAL_PATH = 'evaluate\coco'
ANN_PATH = os.path.join(EVAL_PATH, 'annotations')
# IMG_PATH = os.path.join(EVAL_PATH, 'images')
RES_PATH = os.path.join(EVAL_PATH, 'results')
ann_file_name = 'instances_val2017.json'
ann_file = os.path.join(ANN_PATH, ann_file_name)

coco_gt = COCO(ann_file)

res_file_name = 'val2017_ssd_mobilenet_0.5.json'
res_file = os.path.join(RES_PATH, res_file_name)

coco_dt = coco_gt.loadRes(res_file)

ann_type = 'bbox'

# imgIds = sorted(coco_gt.getImgIds())
# imgIds = imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]

coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
# coco_eval.params.imgIds = imgIds
coco_eval.evaluate()
coco_eval.accumulate()
# coco_eval.summarize()
