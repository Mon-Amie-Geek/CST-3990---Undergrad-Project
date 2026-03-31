import numpy as np

def compute_iou(b1,b2):

    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])

    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])

    inter = max(0, xi2-xi1) * max(0, yi2-yi1)

    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])

    union = a1+a2-inter

    if union == 0:
        return 0

    return inter/union