import torch
import torch.nn as nn
from utils import calculate_iou

class yolo_v3_loss(nn.Module):
    def __init__(self):
        super(yolo_v3_loss, self).__init__()

    def forward(self, pre, label, c=0.5):
        '''
        计算损失，可分为位置损失： center_x, center_y, w, h
                    置信度损失：有物体和没物体损失之和
                    类别损失：

        :param pre: [batch_size, grid, grid, 3, 5 + num_class], (confi, center_x, center_y, w, h)
        :param pre: [batch_size, grid, grid, 3, 5 + num_class], (confi, center_x, center_y, w, h)
        :return:
        '''
        # corr_loss = 0
        # obj_confi_loss = 0
        # noobj_confi_loss = 0
        # class_loss = 0
        # grid_num = label.shape[1]

        batch_size = label.shape[0]

        obj_mask = label[..., 0] == 1
        no_obj_mask = label[..., 0] == 0

        # (x, y, w, h)的loss
        corr_loss = nn.MSELoss()(pre[obj_mask][..., 1:5], label[obj_mask][..., 1:5])
        # 置信度loss
        obj_confi_loss = nn.BCELoss()(pre[obj_mask][..., 0], label[obj_mask][..., 0]) + \
                         0.1 * nn.BCELoss()(pre[no_obj_mask][..., 0], label[no_obj_mask][..., 0])
        # 类别损失
        class_loss = nn.BCELoss()(pre[obj_mask][..., 5:], label[obj_mask][..., 5:])

        return c * corr_loss + (1 - c) * 0.9 * obj_confi_loss + (1 - c) * 0.1 * class_loss


        # 计算13 * 13的损失
        # for i in range(batch_size):
        #     for m in range(grid_num):
        #         for n in range(grid_num):
        #             if label[i, m, n, 0, 0] == 1:
        #                 bbox_1 = (pre[i, m, n, 0, 1] - pre[i, m, n, 0, 3] / 2,
        #                           pre[i, m, n, 0, 2] - pre[i, m, n, 0, 4] / 2,
        #                           pre[i, m, n, 0, 1] + pre[i, m, n, 0, 3] / 2,
        #                           pre[i, m, n, 0, 2] + pre[i, m, n, 0, 4] / 2
        #                           )
        #                 bbox_2 = (pre[i, m, n, 1, 1] - pre[i, m, n, 1, 3] / 2,
        #                           pre[i, m, n, 1, 2] - pre[i, m, n, 1, 4] / 2,
        #                           pre[i, m, n, 1, 1] + pre[i, m, n, 1, 3] / 2,
        #                           pre[i, m, n, 1, 2] + pre[i, m, n, 1, 4] / 2
        #                           )
        #                 bbox_3 = (pre[i, m, n, 2, 1] - pre[i, m, n, 2, 3] / 2,
        #                           pre[i, m, n, 2, 2] - pre[i, m, n, 2, 4] / 2,
        #                           pre[i, m, n, 2, 1] + pre[i, m, n, 2, 3] / 2,
        #                           pre[i, m, n, 2, 2] + pre[i, m, n, 2, 4] / 2
        #                           )
        #
        #                 bbox_gt = (label[i, m, n, 0, 1] - label[i, m, n, 0, 3] / 2,
        #                           label[i, m, n, 0, 2] - label[i, m, n, 0, 4] / 2,
        #                           label[i, m, n, 0, 1] + label[i, m, n, 0, 3] / 2,
        #                           label[i, m, n, 0, 2] + label[i, m, n, 0, 4] / 2
        #                           )
        #
        #                 iou1 = calculate_iou(bbox_1, bbox_gt)
        #                 iou2 = calculate_iou(bbox_2, bbox_gt)
        #                 iou3 = calculate_iou(bbox_3, bbox_gt)
        #
        #                 max_iou = max(iou1, iou2, iou3)

                        # if iou1 == max_iou:
                        #     corr_loss = corr_loss + torch.sum((pre[i, m, n, 0, 1:5] - label[i, m, n, 0, 1:5]) ** 2)
                        #     obj_confi_loss = obj_confi_loss + (pre[i, m, n, 0, 0] - iou1) ** 2
                        #     noobj_confi_loss = noobj_confi_loss + 0.5 *

