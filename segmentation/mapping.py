import numpy as np
import cv2


mapping = {
    'sky': (107.5, 255, 219),
    'water': (122, 255, 112),
    'sand': (54, 255, 255),
    'tree': (0, 0, 0),
    'plant': (144, 255, 255),
    'grass': (18, 255, 255),
    'flower': (90, 255, 255),
    'stone': (72, 255, 255),
    'dirt': (36, 255, 255),
    'snow': (162, 255, 255),
    'unknown': (179, 255, 255)
}

mapping_view = {
    'sky': (191, 215, 255),
    'water': (0, 0, 255),
    'sand': (255, 201, 14),
    'tree': (255, 0, 0),
    'plant': (34, 127, 0),
    'grass': (128, 255, 128),
    'flower': (247, 26, 230),
    'stone': (127, 127, 127),
    'dirt': (139, 69, 19),
    'snow': (255, 255, 255),
    'unknown': (0, 0, 0)
}


def create_readable_mask(mask):
    ref_colors = np.array(list(mapping.values()))
    mask_colored_hsv = ref_colors[mask]
    mask_colored_rgb = cv2.cvtColor(mask_colored_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return mask_colored_rgb
