"""
author: Kwong
time: 2019/8/7 11:05

"""

from torch.autograd import Variable
from torch.utils import data
import numpy as np
from experiment.segmentation.data_loader import loadTestData
from models.efficientnet_pytorch.change_detection import EfficientFPN
import torch
import os
import matplotlib.pyplot as plt
import cv2
import models.swiftnet as swiftnet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
txt_path = 'D:\\download\\segmentation\\test.txt'

di = {1: (0, 200, 0),  # 水      田
      2: (0, 250, 150),  # 水  浇 地
      3: (150, 200, 150),  # 旱  耕 地
      4: (200, 0, 200),  # 园      地
      5: (250, 0, 150),  # 乔木林地
      6: (250, 150, 150),  # 灌木林地
      7: (0, 200, 250),  # 天然草地
      8: (0, 200, 200),  # 人工草地
      9: (0, 0, 200),  # 工业用地
      10: (150, 0, 250),  # 城市住宅
      11: (150, 150, 200),  # 村镇住宅
      12: (150, 150, 250),  # 交通运输
      13: (200, 0, 0),  # 河      流
      14: (200, 150, 0),  # 湖      泊
      15: (250, 200, 0),  # 坑      塘
      0: (0, 0, 0)}  # 其他类别


def test():
    weight_dir = "C:\\Users\\kwong\\Downloads\\w19-43.pkl"
    # model = EfficientFPN.from_name('efficientnet-b4')
    model = swiftnet(pretrained = False, num_classes=16)
    model.load_state_dict(torch.load(weight_dir))
    model.to(0)
    model.eval()
    dataset = loadTestData(txt_path)
    trainloader = data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    with torch.no_grad():
        for i, (images, labels) in enumerate(trainloader):
            wx = images.shape[1]
            wy = images.shape[2]
            l = np.zeros(shape=[wx, wy, 3]).astype(np.uint8)
            step = 512
            blocksize = 1024
            images = np.transpose(images, [0, 3, 1, 2])
            for cx in range(0, wx - step, step):
                for cy in range(0, wy - step, step):
                    nx = cx
                    ny = cy
                    if nx + blocksize > wx:
                        nx = wx - blocksize
                        tx = wx
                    else:
                        tx = cx + blocksize
                    if ny + blocksize > wy:
                        ny = wy - blocksize
                        ty = wy
                    else:
                        ty = cy + blocksize
                    img = images[:, :, nx: tx, ny: ty]  # channel*h*w
                    if img.sum() == 0:
                        continue
                    if torch.cuda.is_available():
                        img = Variable(img.float().cuda())
                    else:
                        img = Variable(img).float()
                    predictions = model(img)
                    pred = torch.argmax(predictions.data[0], dim=0).cpu().numpy()
                    print(np.unique(pred))
                    r = np.zeros(shape=[pred.shape[0], pred.shape[1], 3]).astype(np.uint8)
                    for x in range(pred.shape[0]):
                        for y in range(pred.shape[1]):
                            r[x, y, 0] = di[pred[x, y]][0]
                            r[x, y, 1] = di[pred[x, y]][1]
                            r[x, y, 2] = di[pred[x, y]][2]
                    if nx == 0:
                        if ny == 0:
                            for x in range(0, r.shape[0] - 100):
                                for y in range(0, r.shape[1] - 100):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                        elif ny == wy - blocksize:
                            for x in range(0, r.shape[0] - 100):
                                for y in range(100, r.shape[1]):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                        else:
                            for x in range(0, r.shape[0] - 100):
                                for y in range(100, r.shape[1] - 100):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                    elif ny == 0:
                        if nx != wx - blocksize:
                            for x in range(100, r.shape[0] - 100):
                                for y in range(0, r.shape[1] - 100):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                        else:
                            for x in range(100, r.shape[0]):
                                for y in range(0, r.shape[1] - 100):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                    elif nx == wx - blocksize:
                        if ny == wy - blocksize:
                            for x in range(100, r.shape[0]):
                                for y in range(100, r.shape[1]):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                        else:
                            for x in range(100, r.shape[0]):
                                for y in range(100, r.shape[1] - 100):
                                    l[nx + x, ny + y, 0] = r[x, y, 0]
                                    l[nx + x, ny + y, 1] = r[x, y, 1]
                                    l[nx + x, ny + y, 2] = r[x, y, 2]
                    elif ny == wy - blocksize:
                        for x in range(0, r.shape[0]):
                            for y in range(100, r.shape[1]):
                                l[nx + x, ny + y, 0] = r[x, y, 0]
                                l[nx + x, ny + y, 1] = r[x, y, 1]
                                l[nx + x, ny + y, 2] = r[x, y, 2]
                    else:
                        for x in range(100, r.shape[0] - 100):
                            for y in range(100, r.shape[1] - 100):
                                l[nx + x, ny + y, 0] = r[x, y, 0]
                                l[nx + x, ny + y, 1] = r[x, y, 1]
                                l[nx + x, ny + y, 2] = r[x, y, 2]

            print('done')
            cv2.imwrite('train_{}.tif'.format(i), l)

    pass


if __name__ == '__main__':
    test()
    pass
