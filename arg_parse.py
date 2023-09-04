import argparse

import torch.cuda


def arg_parser():
    label_path = "/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/label.txt"
    img_path = "/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/JPEGImages/"

    parser = argparse.ArgumentParser('yolo V3 cfg')
    parser.add_argument('--cfgfile', default="/home/zxj/lkl_study/CV/yolov2/config-1.yaml",
                        help='net cfgfile path')
    parser.add_argument('--cuda', default=True, help='use cuda')
    parser.add_argument('--device', help='which device')
    parser.add_argument('--label_path', default="/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/label.txt",
                        help='label file path')
    parser.add_argument('--img_path', default="/home/zxj/lkl_study/CV/yolov2/data/VOCdevkit/VOC2007/JPEGImages/",
                        help='img file path')


    args = parser.parse_args()

    if torch.cuda.is_available() and args.cuda:
        args.device = torch.device('cuda:0')

    return args

def arg_parser_HPC():

    parser = argparse.ArgumentParser('yolo V3 cfg')
    parser.add_argument('--cfgfile', default="/share/home/22251009/yolo/yolov3/config-1.yaml",
                        help='net cfgfile path')
    parser.add_argument('--cuda', default=True, help='use cuda')
    parser.add_argument('--device', help='which device')
    parser.add_argument('--label_path', default="/share/home/22251009/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/label.txt",
                        help='label file path')
    parser.add_argument('--img_path', default="/share/home/22251009/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
                        help='img file path')

    args = parser.parse_args()

    if torch.cuda.is_available() and args.cuda:
        args.device = torch.device('cuda:0')

    return args

if __name__ == '__main__':
    args = arg_parser()
    print(args.device)