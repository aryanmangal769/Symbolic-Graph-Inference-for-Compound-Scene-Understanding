import torch
import torch.nn as nn
import numpy as np

def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return [data, target]

def get_bbox_from_segment(annot):
    mask_clean = []
    for mask in annot:
        if len(mask) == 0: continue
        mask = np.array(mask, dtype=np.int32)
        mask_clean.append(mask)
    bbox = get_bbox(mask_clean)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
    return x1, y1, x2, y2

def get_bbox( masks):
    '''
    Get bbox for object masks (1 object may have 1> components). Returns:
    bbox: [x, y, height, width]
    '''
    g_xmin, g_ymin, g_xmax, g_ymax = 10000, 10000, 0, 0
    for mask in masks:
        if len(mask) == 0: continue
        mask = np.array(mask)
        xmin, xmax = np.min(mask[:,0]), np.max(mask[:,0])
        ymin, ymax = np.min(mask[:,1]), np.max(mask[:,1])

        g_xmin = min(g_xmin, xmin)
        g_xmax = max(g_xmax, xmax)
        g_ymin = min(g_ymin, ymin)
        g_ymax = max(g_ymax, ymax)

    bbox = [int(g_xmin), int(g_ymin), int(g_xmax - g_xmin), int(g_ymax - g_ymin)]
    return bbox

def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox #Get the center of the affordance hotspot
    x_center = x1 + (x2 - x1)/2
    y_center = y1 + (y2 - y1)/2
    return x_center, y_center


def get_pixel_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    x_center1, y_center1 = get_bbox_center(bbox1)
    x_center2, y_center2 = get_bbox_center(bbox2)
    return np.sqrt((x_center1 - x_center2) ** 2 + (y_center1 - y_center2) ** 2)
