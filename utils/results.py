import numpy as np
import json

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


def create_json_results_file(
        results,
        file):

    with open(file, 'w') as json_file:
        json.dump(results, json_file)