import numpy as np

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

def draw_labels_and_scores(image, boxes, classes, scores, category_index, label_size):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype('arial.ttf', label_size)
    for i, box in enumerate(boxes):
        text = "{}: {}%".format(category_index[classes[i]]['name'], int(100 * scores[i]))
        font_size = font.getsize(text)
        padding = 2  # 2 pixels

        # Coordinates of the text's background
        bg_left = box[1]
        bg_right = bg_left + font_size[0] + 2 * padding
        bg_top = box[0] - font_size[1] - 2 * padding
        bg_down = box[0]

        # If the upper of the text's background is above the image top, then
        # the label text is displayed below the bounding box, in this case, if the
        # bottom of the text's background is below the image bottom, then the label
        # text is not displayed
        if bg_top < 0:
            bg_down = box[2] + font_size[1] + 2 * padding
            if bg_down < image.shape[0]:
                bg_top = box[2]

                # Coordinates of the top left corner of the text
                text_left = bg_left + padding
                text_top = bg_top + padding

                draw.rectangle([bg_left, bg_top, bg_right, bg_down], fill=COLORS[classes[i]])
                draw.text((text_left, text_top), text, font=font, fill='black')
        else:
            # Coordinates of the top left corner of the text
            text_left = bg_left + padding
            text_top = bg_top + padding

            draw.rectangle([bg_left, bg_top, bg_right, bg_down], fill=COLORS[classes[i]])
            draw.text((text_left, text_top), text, font=font, fill='black')

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
        label_size=18,
        skip_labels_and_scores=True):

    """
    Args:
        image: uint8 numpy array with shape (img_height, img_width, 3).
        boxes: float32 numpy array with shape (N, 4).
        classes: uint8 (default: None) numpy array with shape (N).
        scores: float32 (default: None) numpy array with shape (N).
        category_index: a dict (default: None) containing category dictionaries (each holding
            category index 'id' and category name 'name').
        normalized_coordinates: boolean (default: True) that indicates whether boxes
            should be interpreted as normalized coordinates or not.
        max_boxes_to_draw: integer (default: 20). Maximum number of boxes to visualize.
        min_score_thresh: float (default: 0.5). Minimum score threshold for a box to be visualized.
        line_width: integer (default: 4) controlling line width of the boxes.
        label_size: integer (default: 18) controlling the size of the labels and scores font.
        skip_labels_and_scores: boolean (default: True) that indicates whether to skip
            label and score when drawing a single detection.
    Return:
        uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """

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

        if not skip_labels_and_scores:
            image_np = draw_labels_and_scores(image_np, boxes, classes, scores, category_index, label_size)
    else:
        image_np = image

    return image_np

