import numpy as np
import os
import h5py
import argparse
import xml.etree.ElementTree as ElementTree

parser = argparse.ArgumentParser(description='Conver Pascal VOC 2007 detection dataset to HDF5')
parser.add_argument('-p', '--path_to_voc', help='path to Vocdevkit directory')
sets = ['train', 'val', 'test']
voc_path = ''

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def get_ids(voc_path, name):
    ids = []
    id_path = os.path.join(voc_path, '//%s.txt'%(name))
    print(id_path)
    with open(id_path, 'r') as f:
        ids.extend(f.read().strip().split())
    return ids


def get_img(voc_path, img_id):
    img_path = os.path.join(voc_path, '/%s.jpg'%img_id)
    with open(img_path, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype='uint8')


def get_boxes(voc_path, img_id):
    box_path = os.path.join(voc_path, '/%s.xml'%img_id)
    with open(box_path, 'r') as f:
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
    return np.array(boxes)


def add_to_dataset(voc_path, ids, imgs, boxs):
    for i, img_id in enumerate(ids):
        img_data = get_img(voc_path, img_id)
        img_box = get_boxes(voc_path, img_id)
        imgs[i] = img_data
        boxs[i] = img_box


def main():
    train_ids = get_ids(voc_path, 'train')
    val_ids = get_ids(voc_path, 'val')
    test_ids = get_ids(voc_path, 'test')


    print('Creating HDF5 dataset')
    fname = os.path.join(voc_path, 'voc_2007.hdf5')
    voc_h5file = h5py.File(fname, 'w')
    uint8_dt = h5py.special_dtype(vlen = np.dtype('uint8'))
    int_dt = h5py.special_dtype(vlen = np.dtype(int))

    train_group = voc_h5file.create_group('train')
    val_group = voc_h5file.create_group('val')
    test_group = voc_h5file.create_group('test')

    voc_h5file.attrs['classes'] = np.string_(str.join(',', classes))

    train_imgs = train_group.create_dataset('images', shape=(len(train_ids),), dtype=uint8_dt)
    val_imgs = train_group.create_dataset('images', shape=(len(val_ids),), dtype=uint8_dt)
    test_imgs = train_group.create_dataset('images', shape=(len(test_ids),), dtype=uint8_dt)

    train_boxs = train_group.create_dataset('boxs', shape=(len(train_ids),), dtype=uint8_dt)
    val_boxs = train_group.create_dataset('boxs', shape=(len(val_ids),), dtype=uint8_dt)
    test_boxs = train_group.create_dataset('boxs', shape=(len(test_ids),), dtype=uint8_dt)


    add_to_dataset(voc_path, train_ids,train_imgs, train_boxs)
    add_to_dataset(voc_path, val_ids,val_imgs, val_boxs)
    add_to_dataset(voc_path, test_ids,test_imgs, test_boxs)

    voc_h5file.close()


if __name__ == '__main__':
    main()