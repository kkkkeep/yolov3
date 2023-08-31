import os
from xml.etree import ElementTree as ET
from tqdm import tqdm
CLASS = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

def data_process(anno_path):
    xml_list = os.listdir(anno_path)
    with open("/mnt/data/voc_2007/label.txt", 'w') as f:
        for xml_file_name in tqdm(xml_list):
            data = []
            xml_file = os.path.join(anno_path, xml_file_name)
            et = ET.parse(xml_file)
            root = et.getroot()
            jpg_name = root.find('filename').text
            data.append(jpg_name)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
                obj_class = CLASS.index(obj.find('name').text)
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                xmax = int(bbox.find('xmax').text)
                ymin = int(bbox.find('ymin').text)
                ymax = int(bbox.find('ymax').text)
                center_x, center_y, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin), (ymax - ymin)
                data.extend((obj_class, center_x, center_y, w, h))
            data = [str(i) for i in data]
            f.write(' '.join(data) + '\n')
if __name__ == '__main__':
    annotation_path = "/mnt/data/voc_2007/Annotations/"

    data_process(annotation_path)
    print(len(os.listdir(annotation_path)))
    print(len(CLASS))