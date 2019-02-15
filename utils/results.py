import numpy as np
import os
import json
import lxml.etree as et

def get_results_by_score_threshold(
        image,
        image_id,
        classes,
        boxes,
        scores,
        min_score_thresh=0.5,
        max_detections=20,
        normalized_coordinates=True):

    n_results = np.min(np.where(scores < min_score_thresh))
    n_results = min(n_results, max_detections)

    results = []
    if n_results:
        im_height = image.shape[0]
        im_width = image.shape[1]
        boxes = boxes[:n_results]
        classes = classes[:n_results]
        scores = scores[:n_results]

        if normalized_coordinates:
            boxes[:, 0] = boxes[:, 0] * im_height   # y0
            boxes[:, 1] = boxes[:, 1] * im_width    # x0
            boxes[:, 2] = boxes[:, 2] * im_height   # y1
            boxes[:, 3] = boxes[:, 3] * im_width    # x1
            boxes = np.float32(np.round(boxes, decimals=2))

        for i, box in enumerate(boxes):
            results.append({"image_id": image_id,
                            "category_id": int(classes[i]),
                            "bbox": [float("{0:.2f}".format(box[1])),
                                           float("{0:.2f}".format(box[0])),
                                                 float("{0:.2f}".format(box[3] - box[1])),
                                                       float("{0:.2f}".format(box[2] - box[0]))],
                            "score": float(scores[i])})

        return results
    else:
        return results


def create_results_coco_file(
        results,
        file):

    with open(file, 'w') as json_file:
        json.dump(results, json_file)


def get_pascal_results_by_score_threshold(
        image,
        image_id,
        classes,
        boxes,
        scores,
        index_pascal_categ,
        min_score_thresh=0.5,
        normalized_coordinates=True):
    # only on 20 categories of pascal voc

    pascal_index = [i for i, ind in enumerate(classes) if ind in index_pascal_categ]
    classes = classes[pascal_index]
    boxes = boxes[pascal_index]
    scores = scores[pascal_index]
    n_results = np.min(np.where(scores < min_score_thresh))

    results = []
    if n_results:
        im_height = image.shape[0]
        im_width = image.shape[1]

        boxes = boxes[:n_results]
        classes = classes[:n_results]
        scores = scores[:n_results]

        if normalized_coordinates:
            boxes[:, 0] = boxes[:, 0] * im_height   # y0
            boxes[:, 1] = boxes[:, 1] * im_width    # x0
            boxes[:, 2] = boxes[:, 2] * im_height   # y1
            boxes[:, 3] = boxes[:, 3] * im_width    # x1

        bb_left = boxes[:, 1]
        bb_top = boxes[:, 0]
        bb_right = boxes[:, 3]
        bb_bottom = boxes[:, 2]

        for i, score in enumerate(scores):
            results.append({"image_id": image_id,
                            "category_id": int(classes[i]),
                            "score": score,
                            "bbox": [bb_left[i],bb_top[i], bb_right[i], bb_bottom[i]]})

        return results
    else:
        return results


def create_pascal_result_files_by_categories(
        results,
        category_index,
        path):

    # create a file for each category
    for categ in category_index.values():
        print('Creating results file for the category "%s" ...' % categ['name'])

        file = open(os.path.join(path, '%s.txt' % categ['name']), 'w')

        for r in results:
            if r['category_id'] == categ['id']:
                file.write('%s %.4f %.4f %.4f %.4f %.4f\n' % (r['image_id'], r['score'], r['bbox'][0], r['bbox'][1], r['bbox'][2], r['bbox'][3]))

        file.close()


def save_data_in_xml_file(
        data,
        file_name,
        path):
    """
    Args:
        data: A list of dict with the data to save
        file_name: String
        path: String
    Return:
        A xml file in the 'path' location with the name 'file_name'
    """

    with open(os.path.join(path, '%s.xml' % (file_name)), 'w') as file:
        items = et.Element('items')
        for d in data:
            item = et.SubElement(items, 'item')

            keys = d.keys()
            for key in keys:
                elem = et.SubElement(item, key)
                elem.text = str(d[key])

            text = et.tostring(item, encoding="unicode", pretty_print=True)
            file.write(text)


def read_data_from_xml_file(
        file_name,
        path):

    data = []
    tree = et.parse(os.path.join(path, '%s.xml' % (file_name)))
    root = tree.getroot()
    items = [elem for elem in root.iter() if elem.tag == 'item']
    for item in items:
        dict = {}
        for child in item.getchildren():
            dict[child.tag] = child.text
        data.append(dict)

    return data



