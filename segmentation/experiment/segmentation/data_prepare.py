import os

'''
    为数据集生成对应的txt文件
'''

# path

train_txt_path = 'D:\\download\\segmentation\\train.txt'
label_txt_path = 'D:\\download\\segmentation\\label.txt'
train_dir = 'D:\\download\\segmentation\\train'
test_txt_path = 'D:\\download\\segmentation\\test.txt'
test_dir = 'D:\\download\\segmentation\\test'
valdata_txt_path = 'D:\\download\\segmentation\\valdata.txt'
val_dir = 'D:\\download\\segmentation\\val'
vallabel_txt_path = 'D:\\download\\segmentation\\vallabel.txt'
# valid_txt_path = '../../Data/valid.txt's
# valid_dir = '../../Data/valid/'


def gen_txt(txt_path, img_dir, dtype = 'train'):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        # i_dir = os.path.join(root, _)  # 获取各类的文件夹 绝对路径
        # img_list = os.listdir(i_dir)  # 获取类别文件夹下所有tif图片的路径
        for i in range(len(_)):
            if not _[i].endswith('tif'):  # 若不是tif文件，跳过
                continue
            if dtype == 'train':
                if _[i].find('label') != -1:
                    continue
                else:
                    label = _[i].split('-')[0].split('_')[-1] + '_data'
            else:
                if _[i].find('label') == -1:
                    continue
                else:
                    label = _[i].split('-')[0].split('_')[-1] + '_label'
            # label = sub_dir + '_' + img_list[i].split('_')[-1].split('.')[0]
            img_path = os.path.join(root, _[i])
            path = img_path.replace('\\', '/')  # linux中是正斜杠   windows中是反斜杠 需要替换
            line = path + ' ' + label + '\n'
            f.write(line)

    f.close()


if __name__ == '__main__':
    gen_txt(train_txt_path, train_dir)
    gen_txt(label_txt_path, train_dir, 'label')
    gen_txt(test_txt_path, test_dir)
    gen_txt(valdata_txt_path, val_dir)
    gen_txt(vallabel_txt_path, val_dir, 'label')