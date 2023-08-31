import os
from torchvision.transforms import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
from utils import *


class yolo_v3_dataset(Dataset):
    def __init__(self, label_path, img_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [label.strip() for label in labels]
        self.labels = labels = [label.split(' ') for label in labels]
        self.img_path = img_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        bbox = self.labels[item]
        img_name, bbox = bbox[0], [float(i) for i in bbox[1:]]
        img_file = os.path.join(self.img_path, img_name)
        img, scale_factor = generate_mask(img_file)
        img = transforms.ToTensor()(img)
        for i in range(len(bbox)):
            if i % 5:
                bbox[i] *= scale_factor
        label = bbox2label(bbox)
        # print(label_13.shape)
        # print(label_26.shape)
        # print(label_52.shape)
        return img, label


if __name__ == '__main__':
    # label_path = "/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/label.txt"
    # img_path = "/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/JPEGImages/"


    da = yolo_v3_dataset(label_path, img_path)

    # print(da[3])

    dataloader = DataLoader(da, batch_size=64, shuffle=True, num_workers=4)

    for image, label in dataloader:
        print(image.shape)
        print(type(label))
        break