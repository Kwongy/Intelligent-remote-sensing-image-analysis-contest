
# from __future__ import absolute_import

import argparse
import os
import time
import sys
import torch
from loss.loss import EdgeLoss
from loss.loss import SegmentationLoss
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils import data
from experiment.segmentation.data_loader import loadTrainData
from models.swiftnet import swiftnet
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_path = 'D:\\download\\segmentation\\train.txt'
label_path = 'D:\\download\\segmentation\\label.txt'

def train(args):
    # setup logger
    cur_time = time.localtime()
    log_dir = './logs/' + "logs-{}-{}-{}-{}-{}-{}".format(args.dataset, cur_time.tm_mon, cur_time.tm_mday,
                                                             cur_time.tm_hour, cur_time.tm_min, cur_time.tm_sec)
    # Setup Dataloader
    train_loader = loadTrainData(train_path, label_path, trainforms= True)
    trainloader = data.DataLoader(train_loader, batch_size=args.batch_size, num_workers=0, shuffle=True)
    finetune = True
    freeze_bn = False
    # Setup Model
    weight_dir = "C:\\Users\\kwong\\Downloads\\w19-40.pkl"

    if finetune:
        model = swiftnet(False, num_classes=16)
        weight = torch.load(weight_dir)
        model.load_state_dict(weight, strict=False)
        if freeze_bn:
            for name, param in model.named_parameters():
                if 'bn' in name:
                    param.requires_grad = False
                    print('freeze layer: %s' % name)
    else:
        model = swiftnet(True, num_classes=16)
    model.train()
    if torch.cuda.is_available():
        # model = DataParallelModel(model)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.float()
        model.cuda()
    ce_weight = 0
    iou_weight = 1
    class_criterion = SegmentationLoss(ce_weight=ce_weight, iou_weight=iou_weight)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    writer = SummaryWriter(logdir=log_dir,
                           comment='fine tune on {}, lr {},{} epoch in total'.format(weight_dir, args.l_rate,
                                                                                     args.n_epoch))

    for epoch in range(args.n_epoch):
        loss_sum = 0

        for i, (images, labels) in enumerate(trainloader):
            if torch.cuda.is_available():
                images = Variable(images.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                images = Variable(images).float()
                labels = Variable(labels).float()
            optimizer.zero_grad()
            predictions = model(images)
            loss = class_criterion(predictions, labels)
            segmentation_loss = loss.item()
            writer.add_scalar('global loss', loss.item(), epoch * trainloader.__len__() + i)
            writer.add_scalar('edge_loss', loss.item() - segmentation_loss, epoch * trainloader.__len__() + i)
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum + loss.item()

        mean_loss = loss_sum / len(trainloader)
        print("Epoch [%d/%d] lr: %.7f mean_Loss: %.6f" % (
        epoch + 1, args.n_epoch, optimizer.state_dict()['param_groups'][0]['lr'], mean_loss))
        writer.add_scalar('mean loss', mean_loss, epoch)
        if epoch % 50 == 0:
            torch.save(model.module.state_dict(), os.path.join(log_dir,
                                                               "{}_{}_{}_{}.pkl".format(args.arch, args.dataset,
                                                                                        args.feature_scale,
                                                                                        epoch)))
    torch.save(model.module.state_dict(), os.path.join(log_dir,
                                                       "{}_{}_{}_{}_{}_v.pkl".format(args.arch, args.dataset,
                                                                                     args.batch_size,
                                                                                     args.l_rate,
                                                                                     args.n_epoch)))
    writer.export_scalars_to_json(os.path.join(log_dir, "./all_scalars.json"))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='efficientnet-b4',
                        help='Architecture to use [\'fcn8s, unet, segnet, DFCN etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='Iria',
                        help='Dataset to use [\'pascal, Vaihingen, Potsdam etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=6800,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=7200,
                        help='Height of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=60000,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default= 4,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1,
                        help='Divider for # of features to use')
    args = parser.parse_args()
    train(args)



