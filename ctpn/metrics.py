import Polygon as plg
import csv
import numpy as np

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)

def rectangle_to_polygon(rect):
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(rect.xmin)
    resBoxes[0, 4] = int(rect.ymax)
    resBoxes[0, 1] = int(rect.xmin)
    resBoxes[0, 5] = int(rect.ymin)
    resBoxes[0, 2] = int(rect.xmax)
    resBoxes[0, 6] = int(rect.ymin)
    resBoxes[0, 3] = int(rect.xmax)
    resBoxes[0, 7] = int(rect.ymax)

    pointMat = resBoxes[0].reshape([2, 4]).T

    return plg.Polygon(pointMat)

def rectangle_to_points(rect):
    points = [int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax), int(rect.xmax), int(rect.ymin),
              int(rect.xmin), int(rect.ymin)]
    return points

def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)

def get_intersection_over_union(pD, pG):
    try:
        return get_intersection(pD, pG) / get_union(pD, pG)
    except:
        return 0

def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def match_IOU(pred_bbox_list,gt_bbox_list):
    outputShape = [len(pred_bbox_list),len(gt_bbox_list)]
    iouMat = np.ones(outputShape)*-1
    for i,pred_bbox in enumerate(pred_bbox_list):
        pred_polygon = polygon_from_points(pred_bbox)
        for j,gt_bbox in enumerate(gt_bbox_list):
            gt_polygon = polygon_from_points(gt_bbox)
            IOU_result = get_intersection_over_union(pred_polygon,gt_polygon)
            iouMat[i,j] = IOU_result
    return iouMat

def get_precision(IOU_matrix,threshold=0.5):
    precision_list = np.argmax(IOU_matrix,axis=0)
    precision = np.count_nonzero(precision_list)
    return precision/IOU_matrix.shape[1]


def get_recall(IOU_matrix,threshold=0.5):
    recall_list = np.argmax(IOU_matrix,axis=1)
    recall = np.count_nonzero(recall_list)
    return recall/IOU_matrix.shape[0]
