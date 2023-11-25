import cv2

def visualize_bbox(img, bbox, path ):
    for obj in bbox:
        bbox = obj['object_bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
    cv2.imwrite(path, img)