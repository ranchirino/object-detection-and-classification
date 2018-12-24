import numpy as np
# import tensorflow as tf
# import os
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

# 126 colors, only 126 categories
COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_boxes(image, boxes, classes, line_width):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    for i, box in enumerate(boxes):
        draw.rectangle([box[1], box[0], box[3], box[2]], outline=COLORS[classes[i]], width=line_width)
    return np.array(image_pil)

def visualize_boxes_and_labels(
        image,
        boxes,
        classes=None,
        scores=None,
        category_index=None,
        normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=0.5,
        line_width=4,
        skip_labels=False,
        skip_scores=False):

    total_to_show = np.min(np.where(scores < min_score_thresh))
    total_to_show = min(total_to_show, max_boxes_to_draw)
    image_np = np.array([])
    if total_to_show:
        im_height = image.shape[0]
        im_width = image.shape[1]
        boxes = boxes[:total_to_show]
        classes = classes[:total_to_show]
        scores = scores[:total_to_show]

        if normalized_coordinates:
            boxes[:, 0] = boxes[:, 0] * im_height
            boxes[:, 1] = boxes[:, 1] * im_width
            boxes[:, 2] = boxes[:, 2] * im_height
            boxes[:, 3] = boxes[:, 3] * im_width
            boxes = np.int32(np.round(boxes))

        image_np = draw_boxes(image, boxes, classes, line_width)

        # if not skip_labels:


    return image_np


# scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
# TEST_IMAGE_PATHS = [ os.path.join('test_images', 'image{}.jpg'.format(i)) for i in range(1, 2) ]
#
# image = Image.open(TEST_IMAGE_PATHS[0])
# image = np.array(image)
# image = np.expand_dims(image, 0)
#
# boxes = [[0.2, 0.2, 0.5, 0.5], [0.3, 0.3, 0.6, 0.6]]
# boxes = np.expand_dims(boxes, 0)
#
# tf_img = tf.image.draw_bounding_boxes(image, boxes, name=None)
#
# with tf.Session() as sess:
#     np_img = tf_img.eval()
# plt.imshow(np_img[0])
# plt.show()
