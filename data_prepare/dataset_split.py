import random
import os
"""
将数据划分不同的数据集

"""

def generate_file(file_list, base_path, name):
    file_name = os.path.join(base_path, name)
    # if not os.path.exists(file_name):
    #     os.mkdir(file_name)

    with open(file_name, 'w') as f:
        for i in file_list[:-1]:
            f.write(i + '\n')
        f.write(file_list[-1])

    print(name + ' has been done! ' + str(len(file_list)) + ' files')

def get_name_list(xml_path):
    name_list = []
    img_names = os.listdir(xml_path)

    for name in img_names:
        if name.endswith('.xml'):
            name_list.append(name[:-4])

    return name_list

def get_index(name_list, trainval_ratio, train_ratio):
    N = len(name_list)

    trainval_index = random.sample(range(N), int(N * trainval_ratio))
    train_index = random.sample(trainval_index, int(N * trainval_ratio * train_ratio))

    trainval_list = [name_list[x] for x in trainval_index]
    train_list = [name_list[x] for x in train_index]
    val_list = [x for x in trainval_list if x not in train_list]
    test_list = [x for x in name_list if x not in trainval_list]

    return trainval_list, train_list, val_list, test_list

def main():
    xml_path = "/home/zxj/lkl_study/CV/yolov2/data/" \
               "VOCdevkit/VOC2007/Annotations/"
    base_path = "/home/zxj/lkl_study/CV/yolov2/data/" \
                "VOCdevkit/VOC2007/split_dataset"
    trainval_radio = 0.9
    train_radio = 0.9

    name_list = get_name_list(xml_path)
    trainval_list, train_list, val_list, test_list = \
        get_index(name_list, trainval_radio, train_radio)

    generate_file(trainval_list, base_path, 'trainval.txt')
    generate_file(train_list, base_path, 'train.txt')
    generate_file(val_list, base_path, 'val.txt')
    generate_file(test_list, base_path, 'test.txt')



if __name__ == '__main__':
    main()