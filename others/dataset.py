import numpy as np
import io,os,PIL,h5py,argparse
from PIL import Image
import torch
import torch.utils.data as data

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def get_classes(classes_path):
    with open(classes_path) as f:
        class_name = f.read().strip().split()
    return class_name

def get_anchors(anchors_path):
    if os.path.isfile(anchors_path):
        with open(anchors_path)as f:
            anchors = f.read().strip().split()
        return np.array(list(map(float,anchors))).reshape(-1, 2)
    else:
        Warning('Could not open anchors file, using default.')
        return YOLO_ANCHORS

class yoloDataset(data.Dataset):
    image_size = [416,416]
    def __init__(self,data_path,anchors_path):
        self.anchors = self.get_anchors(anchors_path)
        data = h5py.File(data_path, 'r')
        self.images = data['train/images'][:]
        self.boxes = data['train/boxes'][:]
        # 1 每张图片中，框最多是多少
        self.max_num = 0
        self.num_samples = len(self.boxes)
        self.flag = self.boxes is not None
        if self.flag:
            for i in range(self.num_samples):
                self.boxes[i] = self.boxes[i].reshape(-1,5)
                if self.max_num < self.boxes[i].shape[0]:
                    self.max_num = self.boxes[i].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        processed_images,processed_boxes = self.process_data(idx)
        out = self.encoder(processed_boxes)
        return torch.tensor(processed_images), torch.tensor(out)

    def get_anchors(self,anchors_path):
        if os.path.isfile(anchors_path):
            with open(anchors_path)as f:
                anchors = f.read().strip().split()
            return np.array(list(map(float,anchors))).reshape(-1, 2)
        else:
            Warning('Could not open anchors file, using default.')
            return YOLO_ANCHORS

    def process_data(self,idx):
        '''
        aim :  1.把图片归一化到0`1，转换通道。
               2.box[x1,y1,x2,y2]-->[cx,cy,w,h];在原图上的相对位置;
                 每张图片上框的shape为[max_num,5],多余的补零。
        inputs: idx
        outputs: np.array(img),np.array(new_box)
        '''
        images = self.images[idx]
        boxes = self.boxes[idx]
        img = Image.open(io.BytesIO(images))
        img_shape = np.array(img.size)           #
        img = img.resize(self.image_size, PIL.Image.BICUBIC) #  (416, 416)
        img = np.array(img,np.float)/255.
        img = np.transpose(img,(2,0,1))

        if self.flag:
            box = np.concatenate([(boxes[:,2:4] + boxes[:,:2])*0.5/img_shape,(boxes[:,2:4] - boxes[:,:2])/img_shape,boxes[:,4:5]],1)
            new_box = np.zeros((self.max_num,5),dtype=np.float32)
            new_box[:len(box),:] = box                       # box(cx,cy,w,h,cls)
            return np.array(img),np.array(new_box)
        else:
            return np.array(img),None

    def encoder(self,boxes):
        '''   one picture
        aim   : 把真实框映射到特征图上。
                1. 真实框在特征图上对应的数值；
                2 真实框在特征图上对应的对应的下标；
                3 计算预测偏移
        inputs:
            box[max_num_box, 5(cx,cy,w,h,cls)],anchors[5,2]   max_num_box=10 ; image_size=[416,416]
        outputs:
            true_boxes：[h, w, num_boxes, 4]
            detectors_mask: (h, w, num_boxes, 1)          eg:(13, 13, 5, 1)
            matching_true_boxes:(h, w, num_boxes, 5)      eg:(13, 13, 5, 5)
        '''
        # 1 创建模版
        h,w = self.image_size
        num_anchors = len(self.anchors)
        num_box_params = boxes.shape[1]
        assert h % 32 == 0,'Image sizes in YOLO_v2 must be multiples of 32.'
        assert w % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        grid_h = h//32  # 13
        grid_w = w//32
        true_boxes = np.zeros([grid_h,grid_w,num_anchors,4],dtype=np.float32)
        detectors_mask = np.zeros([grid_h,grid_w,num_anchors,1],dtype=np.float32)  # (13, 13, 5, 1)
        matching_true_boxes = np.zeros([grid_h,grid_w,num_anchors,num_box_params],dtype=np.float32)  # (13, 13, 5, 5)
        # 2 编码
        box_class = boxes[:,4]  # [n,1]
        box = boxes[:,:4]*np.array([grid_w,grid_h,grid_w,grid_h])
        i,j = list(map(int,box[:,0])),list(map(int,box[:,1]))
        best_idx = self.iou_wh(box[:,:2],self.anchors)  #  (10, 2), (5, 2)-->  ((10,), (10,))
        true_boxes[i, j, best_idx] = boxes[:,:4]/np.array([grid_h,grid_w,grid_h,grid_w])
        detectors_mask[i,j,best_idx] = 1
        adjusted_box = np.array(
            [
                box[:,0] - i, box[:,1] - j,
                np.log(box[:,2] / self.anchors[best_idx][:,0]),
                np.log(box[:,3] / self.anchors[best_idx][:,1]), box_class
            ],
            dtype=np.float32).T
        matching_true_boxes[i, j, best_idx] = adjusted_box
        out = np.concatenate([np.array(true_boxes),np.array(detectors_mask),np.array(matching_true_boxes)],-1)
        return out  # true_boxes,detectors_mask, matching_true_boxes  # ((13, 13, 5, 1), (13, 13, 5, 5))

    def iou_wh(self,boxes_wh,anchors_wh):
        '''boxes_wh[n,2],anchors_wh [m,2]
        iou[n,m]'''
        boxes_wh=np.expand_dims(boxes_wh,1)      # [10,1,2]
        anchors_wh=np.expand_dims(anchors_wh,0)  # [1,5,2]
        box_max = boxes_wh/2.
        box_min = -box_max
        anchor_max = anchors_wh/2.
        anchor_min = -anchor_max

        inter_mins = np.maximum(box_min,anchor_min)      # [10,5,2]
        inter_maxs = np.minimum(box_max,anchor_max)
        inter_wh = np.maximum(inter_maxs-inter_mins,0.)
        inter_area = inter_wh[...,0] * inter_wh [...,1]  # [10,5]
        boxes_area = boxes_wh[...,0] * boxes_wh[...,1]
        anchors_area = anchors_wh[...,0]*anchors_wh[...,1]  #[1,5]
        iou = inter_area/(boxes_area+anchors_area-inter_area)  # [10,5]
        best_iou = np.max(iou,1)
        best_idx = np.argmax(iou,1)
        return list(best_idx*(best_iou > 0))

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_path = '../VOCdevkit/pascal_voc_07_12_LS.hdf5'
    anchors_path = '../model_data.pascal_classes.txt'
    train_dataset = yoloDataset(data_path,anchors_path)  # [3, 416, 416],[13, 13, 5, 10]
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0)
    for i,(img,boxes) in enumerate(train_loader):
        print(img.shape)     # torch.Size([1, 3, 416, 416])
        print(boxes.shape)   # torch.Size([1, 13, 13, 5, 10]) 4+1+5

