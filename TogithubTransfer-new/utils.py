import skimage
import skimage.io
import skimage.transform
import numpy as np
import os
from os.path import isfile, isdir
import csv
import glob
import matplotlib.pyplot as plt


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (299, 299), mode='constant')
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx), mode='constant')


def test():
    img = skimage.io.imread("./test_data/starry_night.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx), mode='constant')
    skimage.io.imsave("./test_data/test/output.jpg", img)

def read_pic_name(data_dir):
    contents = os.listdir(data_dir)
    return contents

def make_csv_file(output_csv_path, testA_dir, test_preds):
    testA_files = read_pic_name(testA_dir)
    
    def as_num(x):
        y = '{:.6f}'.format(x) # 6f表示保留6位小数点的float型
        return(y)

    with open(output_csv_path, 'w+', newline='') as csv_file:  
        writer = csv.writer(csv_file)
        
        for index, name in enumerate(testA_files):
            pic_num = name.split('.')[0]
            for count, probability in enumerate(test_preds[index]):
                class_num = count + 1
                class_prob = as_num(probability)
                writer.writerow([pic_num, str(class_num), str(class_prob)])

def run_model(testA_dir, model, pic_class):
    testA_files = read_pic_name(testA_dir)
    test_preds = []
    batch = []
    for ii, file in enumerate(testA_files, 1):
        # 每次添加一张图片
        img = load_image(os.path.join(testA_dir, file))
        # 每次处理一张图片
        batch.append(img)

        # Running the batch through the network to get the codes
        if ii % 100 == 0 or ii == len(testA_files):
            test_preds.append(model.predict(np.array(batch)))
            batch = []
            print('{} images processed'.format(ii))
    return np.array(test_preds).reshape(len(testA_files), pic_class)

def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

if __name__ == "__main__":
    test()
