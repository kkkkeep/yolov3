import numpy as np
import os,h5py,argparse
import xml.etree.ElementTree as ElementTree

sets_from_2007 = [('2007','train'),('2007','val')]
train_set = [('2007','train')]
val_set = [('2007','val')]
test_set = [('2007','test')]

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

parser = argparse.ArgumentParser(description='Conver Pascal VOC 2007 detection dataset to HDF5')
parser.add_argument('-p','--path_to_voc',help='path to VOCdevkit directory',
                    default='../VOCdevkit')

def get_ids(voc_path,datasets):
    ''' 数据集中的样本'''
    ids = []
    for year,set in datasets:
        id_path = os.path.join(voc_path,'VOC%s/ImageSets/Main/%s.txt'%(year,set))
        print(id_path)
        with open(id_path,'r')as f:
            ids.extend(f.read().strip().split())
    return ids

def get_img(voc_path,year,img_id):
    '''  读取图片 '''
    img_path = os.path.join(voc_path,'VOC%s/JPEGImages/%s.jpg'%(year,img_id))
    with open(img_path,'rb')as f:
        data = f.read()
    return np.frombuffer(data,dtype='uint8')  # [n,]

def get_boxes(voc_path,year,img_id):
    '''  读取框 '''
    boxes_path = os.path.join(voc_path,'VOC%s/Annotations/%s.xml'%(year,img_id))
    with open(boxes_path,'r') as f:
        xml_tree = ElementTree.parse(f)
    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        xml_box = obj.find('bndbox')
        bbox = (int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text),
                int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text),
                classes.index(cls))
        boxes.extend(bbox)
    return np.array(boxes)  # [n,]

def add_to_dataset(voc_path,year,ids,images,boxes,start = 0):
    '''  遍历每一个样本，读取数据集的样本和框  '''
    for i,img_id  in enumerate(ids):
        img_data = get_img(voc_path,year,img_id)
        img_box = get_boxes(voc_path,year,img_id)
        images[start+i] = img_data
        boxes[start+i] = img_box
    return i

def _main(args):
    voc_path = os.path.expanduser(args.path_to_voc)
    # 1 获取数据集样本
    train_ids = get_ids(voc_path,train_set)
    val_ids = get_ids(voc_path,val_set)
    test_ids = get_ids(voc_path,test_set)
    train_ids_2007 = get_ids(voc_path,sets_from_2007)
    total_train_ids = len(train_ids)+len(train_ids_2007)

    # 2 设置voc_h5file、数据类型、train_group
    print('Creating HDF5 dataset structure.')
    fname = os.path.join(voc_path,'pascal_voc_07_12_LS.hdf5')
    voc_h5file = h5py.File(fname,'w')
    uint8_dt = h5py.special_dtype(vlen = np.dtype('uint8')) # variable length uint8
    int_dt = h5py.special_dtype(vlen = np.dtype(int))
    train_group = voc_h5file.create_group('train')
    val_group = voc_h5file.create_group('val')
    test_group = voc_h5file.create_group('test')
    # 设置classes，实际应用中没有使用
    voc_h5file.attrs['classes'] = np.string_(str.join(',',classes))
    # 3 设置train_images 、train_boxes容器
    train_images = train_group.create_dataset('images',shape=(total_train_ids,),dtype=uint8_dt)
    val_images = val_group.create_dataset('images',shape=(len(val_ids),),dtype=uint8_dt)
    test_images = test_group.create_dataset('images',shape=(len(test_ids),),dtype=uint8_dt)

    train_boxes = train_group.create_dataset('boxes',shape=(total_train_ids,),dtype=int_dt)
    val_boxes = val_group.create_dataset('boxes',shape=(len(val_ids),),dtype=int_dt)
    test_boxes = test_group.create_dataset('boxes',shape=(len(test_ids),),dtype=int_dt)
    # 4 加载数据
    print('Process Pascal VOC 2007 datasets for training set')
    last_2007 = add_to_dataset(voc_path,'2007',train_ids_2007,train_images,train_boxes)
    print('Processing Pascal VOC 2012 training set.')
    add_to_dataset(voc_path,'2007',train_ids,train_images,train_boxes,start=last_2007+1)
    print('Processing Pascal VOC 2012 val set.')
    add_to_dataset(voc_path, '2007', val_ids, val_images, val_boxes)
    print('Processing Pascal VOC 2007 test set.')
    add_to_dataset(voc_path, '2007', test_ids, test_images, test_boxes)
    print('Closing HDF5 file.')
    voc_h5file.close()
    print('Done.')

if __name__ == '__main__':
    _main(parser.parse_args())
    # voc_path = parser.parse_args().path_to_voc
    # datasets = [('2007','train')]
    # ids = get_ids(voc_path,datasets)
    # # print(ids)
    # img = get_img(voc_path,year='2007',img_id='000025')
    # box = get_boxes(voc_path,year='2007',img_id='000025')
    # print(box.reshape(-1,5))


