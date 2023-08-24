import random,os

xml_path = '../VOCdevkit/VOC2007/Annotations'    # 总样本
base_path = '../VOCdevkit/VOC2007/ImageSets/Main'
trainval_radio = 0.9   # 训练测试数据集的样本比例
train_radio = 0.9      # 验证集比例

names_list = []
img_names = os.listdir(xml_path)
for name in img_names:
    if name.endswith('.xml'):
        names_list.append(name[:-4])

N = len(names_list)       # 总样本量
trainval_num = int(N*trainval_radio)  # 训练测试数据集量
train_num = int(trainval_num*train_radio)  # 训练集样本量
trainval_idx = random.sample(range(N),trainval_num)  # 训练测试数据集下标
train_idx = random.sample(trainval_idx,train_num)
# 训练集下标

# 数据集地址
ftrain_val = open(os.path.join(base_path,'trainval.txt'),'w')
ftrain = open(os.path.join(base_path,'train.txt'),'w')
fval = open(os.path.join(base_path,'val.txt'),'w')
ftest = open(os.path.join(base_path,'test.txt'),'w')

# 读入数据
for i in range(N) :
    name = names_list[i] + '\n'
    if i in trainval_idx:
        ftrain_val.write(name)
        if i in train_idx:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrain_val.close()
ftrain.close()
fval.close()
ftest.close()

