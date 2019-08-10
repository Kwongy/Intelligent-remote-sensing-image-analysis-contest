"""
author: Kwong
time: 2019/8/1 18:28

"""
import sys
import numpy as np
from libtiff import TIFF
from torch.utils.data import Dataset
import random
sys.path.append("..")


def readTif(url):
    tif = TIFF.open(url, mode='r')
    image = tif.read_image()
    return image


def loadTrainData(train_txt_path, label_txt_path, trainforms=False):
    train_data = MyDatasets(txt_path=train_txt_path)
    print('Already load train')
    label_data = MyDatasets(txt_path=label_txt_path, data_type='label')
    print('Already load label')
    createData = []
    dic = {}
    dicl = {}
    for i, (data, name) in enumerate(train_data):
        dic[name] = data
    for i, (data, name) in enumerate(label_data):
        dicl[name] = data
    for key, value in dic.items():
        temp = key.replace('data', 'label')
        data = dicl[temp]
        createData.append([value, data])
    return DealData(dataset=createData, transforms = trainforms)


def loadTestData(test_txt_path):
    test_data = MyDatasets(txt_path=test_txt_path)
    return test_data

class MyDatasets(Dataset):
    def __init__(self, txt_path, data_type='train'):
        imgs = []
        self.imgs = imgs
        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.data_type = data_type
        with open(txt_path, "r") as fh:
            for line in fh:
                line = line.rstrip()
                words = line.split()
                if self.data_type == 'train':
                    imgs.append((words[0] + ' ' + words[1], words[2]))
                else:
                    imgs.append((words[0], words[1]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = readTif(fn)
        if self.data_type == 'train':
            img = img / 255.

        return img, label


class DealData(Dataset):
    def __init__(self, dataset, transforms=False):
        self.imgs = dataset
        self.transforms = transforms
        self.shape = dataset[0][0].shape
        # self.mean = [0.5338931955768501, 0.3687712852153687, 0.3903414877128755, 0.3678001982363022]
        self.mean = [0.3687712852153687, 0.3903414877128755, 0.3678001982363022]
        self.crop_size = 256
        self.dic = {(0, 200, 0): 1,  # 水      田
                    (150, 250, 0): 2,  # 水  浇 地
                    (150, 200, 150): 3,  # 旱  耕 地
                    (200, 0, 200): 4,  # 园      地
                    (150, 0, 250): 5,  # 乔木林地
                    (150, 150, 250): 6,  # 灌木林地
                    (250, 200, 0): 7,  # 天然草地
                    (200, 200, 0): 8,  # 人工草地
                    (200, 0, 0): 9,  # 工业用地
                    (250, 0, 150): 10,  # 城市住宅
                    (200, 150, 150): 11,  # 村镇住宅
                    (250, 150, 150): 12,  # 交通运输
                    (0, 0, 200): 13,  # 河      流
                    (0, 150, 200): 14,  # 湖      泊
                    (0, 200, 250): 15,  # 坑      塘
                    (0, 0, 0): 0  # 其他类别
        }

    def __getitem__(self, index):
        img, label = self.imgs[index]
        # 在这里做transform，转为tensor等等
        if self.transforms is True:
            img, label = self.transform([img, label])
        sp = label.shape
        graph = np.zeros([sp[0], sp[1]], dtype=np.uint8)
        for i in range(sp[0]):
            for j in range(sp[1]):
                temp = (label[i, j, 0], label[i, j, 1], label[i, j, 2])
                graph[i][j] = self.dic[temp]
        label = graph

        return img, label


    def __len__(self):
        return len(self.imgs)

        # data: list [combine_graph, label]

    def transform(self, data):
        data = self.pic_crop(data)  # crop
        # data[0] = self.normalize(data[0])  # std
        data[0] = np.transpose(data[0], (2, 0, 1))
        return data

    def normalize(self, data):
        # print('data.shape : {}'.format(data.shape))
        # print('data before : {}'.format(data))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    data[i, j, k] -= self.mean[k]
        # print('data after : {}'.format(data))
        # print(np.unique(data))
        return data

    def pic_crop(self, pic):
        width1 = random.randint(0, (self.shape[0] - self.crop_size))
        width2 = width1 + self.crop_size
        hight1 = random.randint(0, (self.shape[0] - self.crop_size))
        hight2 = hight1 + self.crop_size
        pic[0] = pic[0][hight1: hight2, width1: width2, :]
        pic[1] = pic[1][hight1: hight2, width1: width2, :]
        return pic

    pass


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    dataset = loadTrainData('D:\\download\\segmentation\\train.txt', 'D:\\download\\segmentation\\label.txt', True)
    print('Already load data')
    imgs = []
    for i, data in enumerate(dataset):
        imgs.append(data)
        print(np.unique(data[1]))
    print(imgs[0][0][:, 77: 88, 77: 88])
    print(imgs[0][1][77: 88, 77 : 88])

    # 6800 * 7200
    # train_path = 'D:\\download\\segmentation\\train.txt'
    # dataset = MyDatasets(train_path)
    # img = []
    # for i, data in enumerate(dataset):
    #     img.append(data[0])
    #
    # sp = img[0].shape
    # mean = [0 for i in range(sp[2])]
    # cnt = 0
    #
    # for i in range(len(img)):
    #     image = img[i]
    #     cnt += 1
    #     for j in range(sp[2]):
    #         for x in range(sp[0]):
    #             for y in range(sp[1]):
    #                 mean[j] += image[x][y][j]
    #
    # for i in range(sp[2]):
    #     mean[i] /= cnt * sp[0] * sp[1]
    #
    # std = [0 for i in range(sp[2])]
    # for i in range(len(img)):
    #     image = img[i]
    #     for j in range(sp[2]):
    #         for x in range(sp[0]):
    #             for y in range(sp[1]):
    #                 std[j] += pow(image[x][y][j] - mean[j], 2)
    #
    # for i in range(sp[2]):
    #     std[i] = np.sqrt(std[i] / cnt)
    # print('mean :' , mean)
    # print('std : ', std)
# mean : [0.5338931955768501, 0.3687712852153687, 0.3903414877128755, 0.3678001982363022]
# std :  [1704.5073034660672, 1682.4791587377583, 1568.8320702720932, 1560.082804542044]
