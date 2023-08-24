import torch
import torch.nn as nn
import yaml
from arg_parse import arg_parser
from utils import *
args = arg_parser()


def create_module(blocks):
    moudle_list = nn.ModuleList()
    in_channels = 3
    output_channels = []
    for index, block in enumerate(blocks):
        moudle = nn.Sequential()
        if block['type'] == 'convolution':
            out_channels = int(block['out_channels'])
            kernel_size = int(block['kernel_size'])
            stride = int(block['stride'])
            padding = int(block['pad'])
            conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size, stride, padding
                             )

            moudle.add_module(f'conv_{index}', conv)
            if block['batch_normalize'] == 1:
                moudle.add_module(f"batch_norm_{index}", nn.BatchNorm2d(out_channels))
            if block['activation'] == 'leaky':
                moudle.add_module(f'activ_{index}', nn.LeakyReLU(0.1))

        elif block['type'] == 'short_cut':
            moudle.add_module(f'short_cut_{index}', empty_layer())

        elif block['type'] == 'route':
            layers = block['layers']
            moudle.add_module(f'route_{index}', empty_layer())
            if isinstance(block['layers'], str):
                layers = layers.split(',')
                layers = [int(a) for a in layers]
                start = layers[0]
                end = layers[1]

                if start > 0:
                    start -= index
                if end > 0:
                    end -= index

                out_channels = output_channels[start] + output_channels[end]
            else:
                if layers > 0:
                    layers -= index
                out_channels = output_channels[layers]

        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=stride)
            moudle.add_module(f'upsample_{index}', upsample)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(a) for a in mask]
            anchors = block['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[2 * i], anchors[2 * i + 1]) for i in range(0, len(anchors) // 2)]
            anchors = [anchors[i] for i in mask]
            detect = detect_layer(anchors)
            moudle.add_module(f'detect_{index}', detect)
        elif block['type'] == 'net_info':
            continue

        else:
            raise ValueError(f'config file {index} block type wrong!')

        moudle_list.append(moudle)
        output_channels.append(out_channels)
        in_channels = out_channels

    return blocks[0], moudle_list


class empty_layer(nn.Module):
    def __init__(self):
        super(empty_layer, self).__init__()

class detect_layer(nn.Module):
    def __init__(self, anchors):
        super(detect_layer, self).__init__()
        self.anchors = anchors

    def forward(self, x, inp_size, num_class, device):
        x = x.data
        prediction = x
        prediction = predict_transform(prediction, inp_size, self.anchors, num_class, device)
        return prediction


def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as f:
        blocks = yaml.load(f.read(), Loader=yaml.FullLoader)
    return blocks


class yolo_v3_net(nn.Module):
    def __init__(self, args):
        super(yolo_v3_net, self).__init__()
        self.blocks = parse_cfg(args.cfgfile)
        self.net_info, self.module_list = create_module(self.blocks)


    def forward(self, x):
        # print(x.shape)
        blocks = self.blocks[1:]
        outputs = []
        write = 0
        detection = []
        for i, block in enumerate(blocks):
            # print(x.shape)
            # print(i)
            # print(block['type'])

            if block['type'] == 'convolution' or block['type'] == 'upsample':
                x = self.module_list[i](x)

            elif block['type'] == 'short_cut':
                _from = block['from']
                x = outputs[-1] + outputs[_from]

            elif block['type'] == 'route':
                layers = block['layers']
                if isinstance(layers, str):
                    layers = layers.split(',')
                    layers = [int(a) for a in layers]
                    if layers[0] > 0:
                        layers[0] -= i

                    if (layers[1] > 0):
                        layers[1] -= i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat([map1, map2], 1)
                else:
                    if layers > 0:
                        layers -= i
                    x = outputs[i + layers]

            elif block['type'] == 'yolo':
                inp_size = int(self.net_info['height'])
                anchors = self.module_list[i][0].anchors
                num_class = block['classes']

                x = predict_transform(x, inp_size, anchors, num_class, args.device)
                detection.append(x)


            elif block['type'] == 'net_info':
                continue

            outputs.append(x)

        try:
            return detection
        except:
            return 0

if __name__ == '__main__':
    args = arg_parser()
    net = yolo_v3_net(args)
    # print(net)
    a = torch.randn((2, 3, 416, 416))
    b = net(a)
