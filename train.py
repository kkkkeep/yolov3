# import torch.optim
from tqdm import tqdm
import torch
from yolo_v3 import yolo_v3_net
from arg_parse import *
from loss_func import yolo_v3_loss
from dataset import yolo_v3_dataset
from torch.utils.data import DataLoader

import logging

def train(net, loss, optimizer, data, device, epochs):
    net.train()

    for epoch in range(0, epochs):
        ll = 0
        for img, label in data:
            img = img.to(device)
            label[0] = label[0].to(device)
            label[1] = label[1].to(device)
            label[2] = label[2].to(device)
            pre = net(img)
            loss_13 = loss(pre[0].float(), label[0].float(), c=0.2)
            loss_26 = loss(pre[1].float(), label[1].float(), c=0.2)
            loss_52 = loss(pre[2].float(), label[2].float(), c=0.2)

            l = loss_13 + loss_26 + loss_52
            ll += l

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        logging.info(f'epoch: {epoch}, loss: {ll}')
        # print(f'epoch: {epoch}, loss: {ll}')
        torch.save(net.state_dict(), f"/share/home/22251009/yolo/yolov3/model/c=0.2_v3.pkl")
        logging.info('Success save')






if __name__ == '__main__':
    args = arg_parser_HPC()
    logging.basicConfig(filename='/share/home/22251009/yolo/yolov3/log/loooing-1.txt', level=logging.DEBUG)
    net = yolo_v3_net(args).to(args.device)
    # net.load_state_dict(torch.load("/home/zxj/lkl_study/CV/yolov2/model/c=0.3_v3.pkl"))
    loss = yolo_v3_loss()

    optimizer = torch.optim.Adam(net.parameters())
    data = yolo_v3_dataset(args.label_path, args.img_path)
    data = DataLoader(data, batch_size=64, shuffle=True, num_workers=4)
    logging.info('batch_size: 64')

    train(net, loss, optimizer, data, args.device, epochs=200)
