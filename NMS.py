import numpy as np

bboxs = np.array([
    [204, 102, 358, 250, 0.5],
    [257, 118, 380, 250, 0.7],
    [280, 135, 400, 250, 0.6],
    [255, 118, 360, 235, 0.7]
])

thresh = 0.3


def nms(bboxs, thresh):
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    scores = bboxs[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 同时计算出四个交区域的(x1, y1, x2, y2)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 可能没有相交区域
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.minimum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = np.where(ovr < thresh)[0]
        # 将下标加一，第一个元素的位置是目标候选框的位置
        order = order[ids + 1]
    return keep


if __name__ == '__main__':
    keep = nms(bboxs, thresh)
    print(keep)
