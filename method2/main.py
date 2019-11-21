# 方案二
# 未使用神经网络
# 详细说明见 idea.jpg
# 具体的实现细节见下

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

PREPROCESS_THRESHOLD = 32
IMG_SIZE = 32

tmp_cnt = 0
tmp_dir = 'tmp'

def preprocess(im):
    global tmp_cnt

    # 转成灰度图
    im = im.convert('L')
    # 统一图片大小
    im = im.resize((IMG_SIZE, IMG_SIZE))
    # 裁剪图片
    w = im.width
    h = im.height
    l = 0
    r = w-1
    u = 0
    b = h-1
    for x in range(1, w-1):
        is_blank = True
        for y in range(1, h-1):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            l = x
            break
    for x in range(w-2, 0, -1):
        is_blank = True
        for y in range(1, h-1):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            r = x
            break
    for y in range(1, h-1):
        is_blank = True
        for x in range(1, w-1):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            u = y
            break
    for y in range(h-2, 0, -1):
        is_blank = True
        for x in range(1, w-1):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            b = y
            break
    im = im.crop((l, u, r, b))
    # 再次统一图片大小
    im = im.resize((IMG_SIZE, IMG_SIZE))
    
    im.save("{}\\{}.png".format(tmp_dir, tmp_cnt))
    tmp_cnt += 1

    # 把图片转成0-1二维数组
    im = np.array(im)
    im[im < PREPROCESS_THRESHOLD] = 0
    im[im >= PREPROCESS_THRESHOLD] = 1
    # plt.imshow(im)
    # plt.show()
    return im

if __name__ == '__main__':
    data_dir = 'data'
    im_prefix = 'data_binary'
    label_file_name = 'label.txt'
    images = []
    labels = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith(im_prefix):
            # 对原始图像做预处理
            im = preprocess(Image.open("{}\\{}".format(data_dir, file_name), 'r'))
            images.append(im)
        elif file_name == label_file_name:
            for l in open('{}\\{}'.format(data_dir, file_name), 'r'):
                labels.append(int(l))
    # print(labels)

