import torch
# import torch.nn as nn
# import yaml
# import numpy as np
# from torchvision import transforms
# import cv2
# import arg_parse
# import yolo_v3
# from utils import *
# from matplotlib import pyplot as plt
# from PIL import Image, ImageDraw


# with open("/home/zxj/lkl_study/CV/yolov2/config.yaml", 'r') as f:
#     data = yaml.load(f.read(), Loader=yaml.FullLoader)
#
# for block in data:
#     if (block['type'] == 'convolution' and block['kernel_size'] == 1):
#         block['pad'] = 0

# with open("/home/zxj/lkl_study/CV/yolov2/config-1.yaml", 'r') as f:
#     data = yaml.load(f.read(), Loader=yaml.FullLoader)
# for i, block in enumerate(data):
#     if block['type'] == 'route':
#         print(block)
# x = np.arange(4)
# a, b = np.meshgrid(x, x)
# print(a)
# print(b)
# x_offset = torch.FloatTensor(a).view(-1, 1)
# y_offset = torch.FloatTensor(b).view(-1, 1)
# print(x_offset)
# print(y_offset)
# x_y_offset = torch.cat((x_offset, y_offset), 1)
# print(x_y_offset)
# x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, 3)
# print(x_y_offset)
# x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, 3).view(-1, 2)
# print(x_y_offset)
# x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, 3).view(-1, 2).unsqueeze(0).unsqueeze(0).view(4, 4, -1, 2)
# print(x_y_offset)
# print(x_y_offset.shape)

# a = torch.arange(18).reshape(3, 6)
# print(a)
# b = a.view(2, 9)
# print(b)

# a = [(1, 2), (3, 4), (5, 6)]
#
# x = torch.FloatTensor(a)
# x = x.repeat(4 * 4, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0).view(1, 4, 4, -1, 2)
# print(x)
# print(x.shape)

# a = torch.arange(6).reshape(2, 3)
# print(a)
# a = a.repeat(2, 2)
# print(a)
# a = torch.arange(6).reshape(2, 3)
# print(a)
# #tensor([[0, 1, 2],
# #       [3, 4, 5]])
# a = a.view(-1, 2)
# print(a)

# pre = torch.tensor([0.8], dtype=torch.float32)
# print(torch.log(pre))
# label = torch.tensor([1], dtype=torch.float32)
# loss = nn.BCELoss()

# l = loss(pre, label)
# print(l)

# # pre = torch.tensor([0.8, 0.8, 0.8, 0.8] * 24).reshape((4, 4, 6))
# # label = torch.ones((4, 4, 6))
# #
# # loss = nn.BCELoss(reduction='none')
# # a = loss(pre[..., 1:5], label[..., 1:5])
# #
# # print(a)
#
# # print(pre)
# # print(label.shape)
#
# test_path = "/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/JPEGImages/009175.jpg"
# args = arg_parse.arg_parser()
#
# in_img, scale_factor = generate_mask(test_path)
# print(type(in_img))
# # plt.imshow(in_img)
# # plt.show()
#
# in_img_1 = transforms.ToTensor()(in_img).unsqueeze(0).to(args.device)
#
# net = yolo_v3.yolo_v3_net(args)
#
# net.load_state_dict(torch.load("/home/zxj/lkl_study/CV/yolov2/model/c=0.3_v3.pkl"))
#
# net.to(args.device)
#
# pre = net(in_img_1)
#
# mask_1 = pre[0][..., 0] > 0.95
# mask_2 = pre[1][..., 0] > 0.97
# mask_3 = pre[2][..., 0] > 0.97
#
# # print(mask_1.shape)
# # print(pre[0][mask_1].shape)
# # print(pre[0][mask_1][:, 1:5] / scale_factor)
# out = pre[0][mask_1][:, 1:5] / scale_factor
# out = out.cpu().detach().numpy()
# print(out.shape)
# b = np.empty_like(out)
# b[:, 0] = out[:, 0] - out[:, 2] / 2
# b[:, 1] = out[:, 1] - out[:, 3] / 2
# b[:, 2] = out[:, 0] + out[:, 2] / 2
# b[:, 3] = out[:, 1] + out[:, 3] / 2
# # img = cv2.imread(test_path)
# a = ImageDraw.ImageDraw(in_img)
#
# for i in range(30, 40):
#     x1 = int(b[i, 0])
#     y1 = int(b[i, 1])
#     x2 = int(b[i, 2])
#     y2 = int(b[i, 3])
#     # print(x1, y1, x2, y2)
#     a.rectangle(((x1, y1), (x2, y2)), outline='red', width=2)
# plt.imshow(in_img)
# plt.show()
#     cv2.rectangle(img, pt1 = (x1, y1), pt2=(x2, y2), color=(155, 155, 155), thickness=2)
# cv2.imwrite("/home/zxj/lkl_study/CV/yolov2/data/test.jpg", img)
# print(pre[1][mask_2].shape)
# print(pre[2][mask_3].shape)
#
# corr_1 = pre[0][mask_1][:, 1:5]
# corr_2 = pre[1][mask_2][:, 1:5]
# corr_3 = pre[2][mask_3][:, 1:5]
#
# print(corr_1.shape)
# print(corr_2.shape)
# print(corr_3.shape)

# print(pre[0].shape)
# print(pre[1].shape)
# print(pre[2].shape)

print(torch.cuda.is_available())
