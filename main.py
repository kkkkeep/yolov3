import torch
import torch.nn as nn
import yaml
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='test model by config file')
    parser.add_argument('--config_file', default="/home/zxj/lkl_study/CV/yolov2/config.yaml",
                        type=str, help='position of config file')
    return parser.parse_args()

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


if __name__ == '__main__':
    args = arg_parse()
    with open(args.config_file, 'r') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    moudle_list = nn.ModuleList()
    in_channels = 3
    output_channels = []

    for index, block in enumerate(result):
        moudle = nn.Sequential()
        if block['type'] == 'convolution':
            out_channels = block['out_channels']
            kernel_size = block['kernel_size']
            stride = block['stride']
            if block['pad'] == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            moudle.add_module(f'conv_{index}', conv)
            if block['batch_normalize'] == 1:
                moudle.add_module(f'batch_norm_{index}', nn.BatchNorm2d(out_channels))
            if block['activation'] == 'leaky':
                moudle.add_module(f'activ_{index}', nn.LeakyReLU(0.1, inplace=True))

        elif block['type'] == 'short_cut':
            moudle.add_module(f'short_cut_{index}', EmptyLayer())

        elif block['type'] == 'route':
            moudle.add_module(f'route_{index}', EmptyLayer())
            block['layers'] = block['layers'].split(',')
            start = block['layers'][0]
            try:
                end = block['layer'][1]
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            if end < 0:
                out_channels = output_channels[index + start] + out_channels[index + end]
            else:
                out_channels = output_channels[index + start]

        elif block['type'] == 'upsample':
            stride = block['stride']
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            moudle.add_module(f'upsample_{index}', upsample)

        elif block['type'] == 'net_info':
            continue

        elif block['type'] == 'yolo':


        else:
            raise ValueError(f'config file {index} block type wrong!')


        moudle_list.append(moudle)
        in_channels = out_channels
        output_channels.append(out_channels)


    print(moudle_list)

    # a = torch.randn((1, 3, 48, 48))
    # print(a.shape)

    # for block in moudle_:
    #     a = block(a)
    #     print(a.shape)
    # a = moudle(a)
    # print(a.shape)