import numpy as np
import torch
from PIL import Image
# import matplotlib.pyplot as plt

def predict_transform(prediction, inp_size, anchors, num_class, device):
    '''
    将predict转换为便于计算loss的形式
    predict中，center_x, center_y, w, h是相对于一个grid的值，
    返回的是在原始输入图片上的值
    :param prediction: 预测值 [batch, B * (1 + 4 + num_class), num_grid, num_grid]
    :param inp_size: 原始输入图像的size
    :param anchors: 当前detect层的anchor框的大小, 一共B个
    :param num_class: 一共有多少类别数
    :param device: 所用设备
    :return: [batch, grid * grid * B, 1 + 4 + num_class]
    '''
    batch_size = prediction.size(0)
    bbox_attr = 5 + num_class
    num_anchors = len(anchors)
    grid_num = prediction.size(2)
    stride = inp_size // grid_num
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # prediction = prediction.view(batch_size, num_anchors * bbox_attr, grid_num * grid_num)\
    #                 .permute(0, 2, 1)\
    #                 .contiguous()
    # prediction = prediction.view(batch_size, grid_num * grid_num * num_anchors, bbox_attr)

    prediction = prediction.permute(0, 2, 3, 1).contiguous()
    prediction = prediction.view(batch_size, grid_num, grid_num, 3, -1)


    prediction[..., 0] = torch.sigmoid(prediction[..., 0])
    prediction[..., 1] = torch.sigmoid(prediction[..., 1])
    prediction[..., 2] = torch.sigmoid(prediction[..., 2])

    grid_len = np.arange(grid_num)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    x_y_offset = torch.cat([x_offset, y_offset], 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0).unsqueeze(0).unsqueeze(0).view(1, grid_num, grid_num, -1, 2).to(device)
    # print(prediction.shape)
    # print(x_y_offset.shape)

    prediction[..., 1:3] += x_y_offset

    anchors = torch.FloatTensor(anchors)
    anchors = anchors.repeat(grid_num * grid_num, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0).view(1, grid_num, grid_num, -1, 2).to(device)
    prediction[..., 3:5] = torch.exp(prediction[..., 3:5]) * anchors

    prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])

    prediction[..., 1:5] *= stride

    return prediction

def generate_mask(img_path):
    img = Image.open(img_path)
    h, w = img.size
    size = max(h, w)
    mask = Image.new(mode='RGB', size=(size, size), color=(0, 0, 0))
    # plt.imshow(mask)
    mask.paste(img, box=(0, 0))
    mask = mask.resize((416, 416))

    return mask, 416 / size

def _bbox2label(bbox, grid_num):
    label = np.zeros((grid_num, grid_num, 3, 5 + 20))
    strid = 416 // grid_num
    for i in range(len(bbox) // 5):
        x_grid = int(bbox[5 * i + 1] // strid)
        y_grid = int(bbox[5 * i + 2] // strid)
        for j in range(3):
            label[y_grid, x_grid, j, 0] = 1
            label[y_grid, x_grid, j, 1:5] = np.array(bbox[5 * i + 1 : 5 * i + 5])
            label[y_grid, x_grid, j, 5 + int(bbox[5 * i])] = 1
    return torch.from_numpy(label)

def bbox2label(bbox):
    '''
    将[class, x, y, w, h]的bbox转化为(grid_num * grid_num * num_anchors, 5 + num_class)的形式
    :param bbox: [class, x, y, w, h ...]
    :return:
    '''
    assert len(bbox) % 5 == 0, "bbox的长度为5的整数倍"
    label_13 = _bbox2label(bbox, 13)
    label_26 = _bbox2label(bbox, 26)
    label_52 = _bbox2label(bbox, 52)

    return [label_13, label_26, label_52]

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    _x1 = max(x1, x3)
    _y1 = max(y1, y3)
    _x2 = max(x2, x4)
    _y2 = max(y2, y4)
    inter_area = (_x2 - _x1) * (_y2 - _y1)
    union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter_area

    return inter_area / union_area



if __name__ == '__main__':
    generate_mask('/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/JPEGImages/000964.jpg')