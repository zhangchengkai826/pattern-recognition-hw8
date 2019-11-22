# 方案二
# 未使用神经网络
# 详细说明见 idea.jpg
# 具体的实现细节见下

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

PREPROCESS_THRESHOLD = 32
IMG_SIZE = 96
BORDER = 5

tmp_cnt = 0
tmp_dir = 'tmp'

DIVISION = 0.2
ONE_RATIO_THRESHOLD = 0.5

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
    for x in range(BORDER, w-BORDER):
        is_blank = True
        for y in range(BORDER, h-BORDER):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            l = x
            break
    for x in range(w-BORDER-1, BORDER-1, -1):
        is_blank = True
        for y in range(BORDER, h-BORDER):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            r = x
            break
    for y in range(BORDER, h-BORDER):
        is_blank = True
        for x in range(BORDER, w-BORDER):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            u = y
            break
    for y in range(h-BORDER-1, BORDER-1, -1):
        is_blank = True
        for x in range(BORDER, w-BORDER):
            if im.getpixel((x, y)) < PREPROCESS_THRESHOLD:
                is_blank = False
        if not is_blank:
            b = y
            break
    im = im.crop((l, u, r, b))
    # 再次统一图片大小
    im = im.resize((IMG_SIZE, IMG_SIZE))
    # 把图片旋转90度
    im = im.rotate(90)
    
    im.save("{}\\{}.png".format(tmp_dir, tmp_cnt))
    tmp_cnt += 1

    # 把图片转成0-1二维数组
    im = np.array(im)
    im[im < PREPROCESS_THRESHOLD] = 0
    im[im >= PREPROCESS_THRESHOLD] = 1
    # plt.imshow(im)
    # plt.show()
    return im

# 对图片第[s, e]行进行特征提取
# 出现只有一组连续点的行数超过50%，返回1
# 出现只有一组连续点的行，但只有一组连续点的行数未超过50%，返回2
# 未出现只有一组连续点的行，返回3
# 注意：如果有一行没有出现任何连续点，则不纳入行数统计
# 若将图像划分为上中下三部分（中间部分较大，上下部分较小）
# 则理想状况下：
# 勾 - [1 1 1], 叉 - [1/3 2 1/3], 圈 - [2/3 3 2/3]
# 所以只需要看特征向量的第2个分量即可
# 此时，分类准确率为76.54%
# 错误分类信息详见error.log
# 增大预处理后的图像尺寸（32x32 -> 64x64），分类准确率提升至91.36%
# 再次增大预处理后的图像尺寸（64x64 -> 96x96），分类准确率提升至95.06%
# 再次增大预处理后的图像尺寸（96x96 -> 128x128），分类准确率反而降低了，降至92.59%
# 再次增大预处理后的图像尺寸（128x128 -> 256x256），分类准确率再次降低，降至86.42%
def extrack_feature(im, s, e):
    total = e-s+1
    one = 0
    for y in range(s, e+1):
        cnt = 0
        met_black = False
        for x in range(IMG_SIZE):
            if not met_black:
                if im[y][x] == 0:
                    met_black = True
            else:
                if im[y][x] == 1:
                    met_black = False
                    cnt += 1
        if met_black:
            cnt += 1
        
        if cnt == 0:
            total -= 1
        elif cnt == 1:
            one += 1

    if total == 0 or one == 0:
        return 3
    ratio = one / total
    if ratio > ONE_RATIO_THRESHOLD:
        return 1
    else:
        return 2

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
    
    dh = int(IMG_SIZE * DIVISION)
    idx = 0
    correct = 0
    for im in images:
        # 特征提取
        v = []
        v.append(extrack_feature(im, 0, dh-1))
        v.append(extrack_feature(im, dh, IMG_SIZE-dh-1))
        v.append(extrack_feature(im, IMG_SIZE-dh, IMG_SIZE-1))

        # 识别分类统计
        # 0 - 勾, 1 - 叉，2 - 圈
        mapping = [None, 0, 1, 2]
        prediction = mapping[v[1]]

        if prediction == labels[idx]:
            correct += 1
        else:
            print("Features: {}".format(v))
            print("Prediction: {}, Label: {}".format(prediction, labels[idx]))
            # plt.imshow(im)
            # plt.show()
        idx += 1
    print("Accuracy: {0:.2f}%".format(correct / len(labels) * 100))
