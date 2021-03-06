import numpy as np
import os
import lxml.etree as et
import matplotlib.pyplot as plt

# get detection annotations
def get_gt_annot(
        path,
        det_files_id,
        difficult=False):
    """
    Args:
        path: string path of annotation files
        det_files_id: list of string of detection images id
        difficult: boolean (default: False) if the annotations of the difficult objects are obtained.
    Return:
        list of dict that contains the image id (string), the category (string),
        the bounding box coordinates (float) of the ground truth annotations, and the difficult annotation (boolean)
    """

    annot = []
    i = 0
    n_files = len(det_files_id)
    for file in os.listdir(path):
        image_id = str(file).replace('.xml', '')

        if image_id in det_files_id:
            i += 1
            print('Reading file %d of %d' % (i, n_files))
            tree = et.parse(os.path.join(path, file))
            root = tree.getroot()
            objects = [elem for elem in root.iter() if elem.tag == 'object']
            for obj in objects:
                if (obj.findtext('difficult') == '0') or (obj.findtext('difficult') == '1' and difficult):
                    categ = obj.findtext('name')
                    bbox = [c for c in obj.iter() if c.tag == 'bndbox']
                    left = bbox[0].findtext('xmin')
                    top = bbox[0].findtext('ymin')
                    right = bbox[0].findtext('xmax')
                    bottom = bbox[0].findtext('ymax')
                    diffic = np.int(obj.findtext('difficult'))

                    annot.append({'image_id': image_id,
                                  'category': categ,
                                  'bb_left': np.float(left),
                                  'bb_top': np.float(top),
                                  'bb_right': np.float(right),
                                  'bb_bottom': np.float(bottom),
                                  'difficult': np.bool(diffic)})

    return annot


def get_det_results_by_category_files(
        path,
        categories_list):
    """
    Args:
        path:  string path of txt result files
        categories_list: list of string categories
    Return:
        list of dict that contains the category, the image id (string), the confidence score (float),
        and the bounding box coordinates (float) of the detection results
    """
    results = []
    for file_name in os.listdir(path):
        if str(file_name).endswith('.txt'):
        # if '.txt' in str(file):
            categ = str(file_name).replace('.txt', '')
            if categ in categories_list:
                print('%s: loading results' % (categ))
                with open(os.path.join(path, file_name)) as file:
                    # item = file.readline()
                    for line in file.readlines():
                        item = line.replace('\n','')
                        res = item.split(' ')
                        results.append({'category': categ,
                                        'image_id': res[0],
                                        'score': np.float(res[1]),
                                        'bb_left': np.float(res[2]),
                                        'bb_top': np.float(res[3]),
                                        'bb_right': np.float(res[4]),
                                        'bb_bottom': np.float(res[5])})

    return results


def obtain_tp_and_fp_by_category(
        gt_list,
        det_list,
        image_ids,
        iou_threshold=0.5):
    """
    Args:
        gt_list: list of ground truth for a specific category X
        det_list: list of detections for a specific category X
        iou_threshold: float value that indicates the threshold of the true positives detections
        image_ids: list of string image ids
    Return:
        tp: numpy vector of length(det_list) that indicates (with 1) which detections are true positives
        fp: numpy vector of length(det_list) that indicates (with 1) which detections are false positives. fp = not(tp)
    """

    # tp = np.zeros((len(det_list),))
    # fp = np.zeros((len(det_list),))
    #
    # # assign detection to ground truth object
    # for i, det in enumerate(det_list):
    #     image_id = det['image_id']
    #
    #     # obtain all ground truth of this image
    #     image_gt = [gt for gt in gt_list if gt['image_id'] == image_id]
    #
    #     # iou calculation of the detection with the ground truth
    #     max_iou = 0
    #     for gt in image_gt:
    #         # calculation of the intersection bounding box
    #         inters_bb = [max(gt['bb_left'], det['bb_left']),
    #                     max(gt['bb_top'], det['bb_top']),
    #                     min(gt['bb_right'], det['bb_right']),
    #                     min(gt['bb_bottom'], det['bb_bottom'])]
    #
    #         inters_width = float(inters_bb[2]) - float(inters_bb[0]) + 1
    #         inters_height = float(inters_bb[3]) - float(inters_bb[1]) + 1
    #
    #         if inters_width > 0 and inters_height > 0:
    #             # there is intersection
    #             inters_area = inters_width * inters_height
    #             gt_area = (float(gt['bb_right']) - float(gt['bb_left']) + 1) * (float(gt['bb_bottom']) - float(gt['bb_top']) + 1)
    #             det_area = (float(det['bb_right']) - float(det['bb_left']) + 1) * (float(det['bb_bottom']) - float(det['bb_top']) + 1)
    #             union_area = gt_area + det_area - inters_area
    #             iou = inters_area / union_area
    #             if iou > max_iou:
    #                 max_iou = iou
    #
    #     if max_iou >= iou_threshold:
    #         tp[i] = 1
    #     else:
    #         fp[i] = 1

    #################################################################
    det_tp_ann = []  # detections with tp annotation

    for id in image_ids:
        # obtain all ground truth of this image
        image_gt = [gt for gt in gt_list if gt['image_id'] == id]
        if len(image_gt):
            # obtain all detections of this image
            image_det = [d for d in det_list if d['image_id'] == id]

            if len(image_det):
                # if there are detections in this image
                # add tp annotation
                for d in image_det:
                    d['tp'] = 0

                for gt in image_gt:
                    # calculo todos los iou con este gt y veo si el mayor valor es mayor que el umbral de iou
                    iou = []
                    for d in image_det:
                        iou.append(compute_iou_area(gt,d))

                    # pos_max_iou = np.argmax(iou)
                    if max(iou) >= iou_threshold:
                        # index = np.argmax(iou)
                        # if len(index) > 1:
                        #
                        image_det[np.argmax(iou)]['tp'] = 1

                for d in image_det:
                    det_tp_ann.append(d)
        else:
            false_det = [d for d in det_list if d['image_id'] == id]
            for d in false_det:
                d['tp'] = 0
                det_tp_ann.append(d)

    tp = np.array([d['tp'] for d in det_tp_ann])
    fp = 1-tp

    return det_tp_ann, tp, fp


def compute_iou_area(
        gt,
        det):

    inters_bb = [max(gt['bb_left'], det['bb_left']),
                 max(gt['bb_top'], det['bb_top']),
                 min(gt['bb_right'], det['bb_right']),
                 min(gt['bb_bottom'], det['bb_bottom'])]

    inters_width = inters_bb[2] - inters_bb[0] + 1
    inters_height = inters_bb[3] - inters_bb[1] + 1

    iou = 0
    if inters_width > 0 and inters_height > 0:
        # there is intersection
        inters_area = inters_width * inters_height
        gt_area = (gt['bb_right'] - gt['bb_left'] + 1) * (gt['bb_bottom'] - gt['bb_top'] + 1)
        det_area = (det['bb_right'] - det['bb_left'] + 1) * (det['bb_bottom'] - det['bb_top'] + 1)
        union_area = gt_area + det_area - inters_area
        iou = inters_area / union_area

    return iou



def compute_precision_and_recall(
        tp,
        fp,
        n_gt):
    """
    Args:
        tp: numpy vector of length(det_list) that indicates (with 1) which detections are true positives
        fp: numpy vector of length(det_list) that indicates (with 1) which detections are false positives. fp = not(tp)
        n_gt: int that indicates the number of ground truth
    Return:
        precision: numpy vector of length = length(tp)
        recall: numpy vector of length = length(tp)
    """

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / n_gt
    precision = tp / (tp + fp)

    return precision, recall


def plot_precision_recall_curve(
        prec,
        recall,
        categ,
        ap):
    plt.plot(recall, prec)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('Precision/recall curve')
    plt.legend(['%s: AP = %.2f' % (categ, ap)], loc='best')
    plt.axis([0,1,0,1])
    plt.grid(True)
    plt.show()


def interpolated_average_precision(
        precision,
        recall):
    m_rec = np.concatenate(([0], recall))
    m_pre = np.concatenate((precision, [0]))

    for i in range(len(m_pre)-2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i+1])
    m_pre = m_pre[:-1]

    change_rec = m_rec[1:] - m_rec[:-1]
    ap = sum(change_rec * m_pre)
    # i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    # ap = sum((m_rec[i] - m_rec[i-1]) * m_pre[i])

    return ap

def average_precision(
        precision,
        recall):
    m_rec = np.concatenate(([0], recall))
    change_rec = m_rec[1:] - m_rec[:-1]
    ap = sum(change_rec * precision)

    return ap


def mean_average_precision(avg_prec):
    sum_cum = 0
    for categ in avg_prec:
        sum_cum += categ['ap']

    return sum_cum / len(avg_prec)


def save_ap_by_category(
        avg_prec,
        file_name,
        path):
    """
    Args:
        avg_prec: list of dict that contains the keys 'category' and their 'ap' value.
        file_name: string that indicates the name of the resulting file.
        path: string of the location to save the file.
    Return:
        An xml file with a saved values of ap by category.
    """

    with open(os.path.join(path, '%s.xml' % (file_name)), 'w') as file:
        for c in avg_prec:
            item = et.Element('item')

            category = et.SubElement(item, 'category')
            category.text = c['category']

            ap = et.SubElement(item, 'ap')
            ap.text = str(c['ap'])

            data = et.tostring(item, encoding="unicode", pretty_print=True)
            file.write(data)











# file = open(os.path.join(RES_PATH_PASCAL, 'items.xml'), "w")
# total_items = len(total_results)
# for i, r in enumerate(total_results):
#     print('Writing item %d of %d' % (i+1, total_items))
#     item = et.Element('item')
#
#     image_id = et.SubElement(item, 'image_id')
#     image_id.text = r['image_id']
#
#     category_id = et.SubElement(item, 'category_id')
#     category_id.text = str(r['category_id'])
#
#     score = et.SubElement(item, 'score')
#     score.text = str(r['score'])
#     #
#     bbox = et.SubElement(item, 'bbox')
#     bb_left = et.SubElement(bbox, 'bb_left')
#     bb_top = et.SubElement(bbox, 'bb_top')
#     bb_right = et.SubElement(bbox, 'bb_right')
#     bb_bottom = et.SubElement(bbox, 'bb_bottom')
#     bb_left.text = str(r['bbox'][0])
#     bb_top.text = str(r['bbox'][1])
#     bb_right.text = str(r['bbox'][2])
#     bb_bottom.text = str(r['bbox'][3])
#
#     data = et.tostring(item, encoding="unicode", pretty_print=True)
#     file.write(data)
# file.close()