import tensorflow as tf
from scipy.ndimage import imread
import os
from os.path import isfile, isdir
from scipy import misc
import random
import numpy as np


def fix_gpu_error():
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def preprocess_all_pics(src_data_dir, dst_data_dir, target_size = [224, 224], random_crop_window = [448, 448], data_plus = 10, train = False):
    classes = read_pic_path(src_data_dir)
    for pic_class in classes:
        print(src_data_dir + pic_class + " -> " + dst_data_dir + pic_class  + "#  start process")
        preprocess_pics(os.path.join(src_data_dir, pic_class), os.path.join(dst_data_dir, pic_class), target_size, random_crop_window, data_plus, train)

# 只增强 训练 数据集
def preprocess_pics(src_data_dir, dst_data_dir, target_size, random_crop_window, data_plus=10, train = False):
    fix_gpu_error()
    if not os.path.exists(dst_data_dir):
        os.makedirs(dst_data_dir)
        files = os.listdir(src_data_dir)
        for ii, file in enumerate(files, 1):
            _preprocess_single_pic(os.path.join(src_data_dir, file), os.path.join(dst_data_dir, file), target_size, random_crop_window, data_plus, train)
            # 定时释放资源
            tf.reset_default_graph()
            if ii % 10 == 0 or ii == len(files):    
                print("{} images processed".format(ii))

# 生成子文件夹列表
def read_pic_path(data_dir):
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]
    return classes

def _preprocess_single_pic(src_path,  dst_path, target_size, random_crop_window, data_plus, train):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = imread(src_path)

        if train == True:
            for i in range(data_plus):
                # 随机拉伸，随机裁剪，随机亮度，随机旋转
                #处理尺寸到不拉伸的正方形
                processed_img = random_process_image(img, target_size, random_crop_window)
                #处理尺寸到不拉伸的 target_size
                processed_img = tf.image.resize_images(processed_img, [target_size[0], target_size[1]], method = 1)
                encoded_image = tf.image.encode_jpeg(processed_img)
                with tf.gfile.GFile(dst_path + "ex" + str(i) + ".jpg", "wb") as f:
                    f.write(encoded_image.eval())
        else:
            #非训练集 只调整尺寸
            # 调整到不拉伸的正方形（裁剪长边，损失部分图像）
            resize = min(img.shape[0], img.shape[1])
            resize = max(resize, target_size[0]) 
            processed_img = tf.image.resize_image_with_crop_or_pad(img, resize, resize)
            processed_img = tf.image.resize_images(processed_img, [target_size[0], target_size[1]], method = 1)
            encoded_image = tf.image.encode_jpeg(processed_img)
            with tf.gfile.GFile(dst_path + ".jpg", "wb") as f:
                f.write(encoded_image.eval())
            # 调整到不拉伸的正方形（填充短边，缩小了分辨率）
            #resize = max(img.shape)
            #img = tf.image.resize_image_with_crop_or_pad(img, resize, resize)
            #img = tf.image.resize_images(img, [224, 224], method = 1)
            #encoded_image = tf.image.encode_jpeg(img)
            #with tf.gfile.GFile(dst_path + ".jpg", "wb") as f:
            #   f.write(encoded_image.eval())



### 图像数据增强

def random_process_image(image, target_size, random_crop_window):
    # 随机拉伸
        # 获得两个随机数
        # 进行水平或者垂直拉伸
#    processed_img = random_stretching_process(image)
#    h = processed_img.eval().shape[0]
#    w = processed_img.eval().shape[1]
    h  = image.shape[0]
    w = image.shape[1]
    if ((h > random_crop_window[0]) and (w > random_crop_window[1])):
        processed_img = random_size_process(image, random_crop_window)
    else:
        resize = min(h, w)
        resize = max(resize, target_size[0])
        processed_img = tf.image.resize_image_with_crop_or_pad(image, resize, resize)

#    processed_img = tf.image.convert_image_dtype(processed_img, dtype=tf.float32)
#    processed_img = random_color_process(processed_img, color_ordering = 1)
#    processed_img = random_pos_process(processed_img)
    return processed_img

def random_stretching_process(image):
    if 1 == random.randint(0, 1):
        h_streching = random.randint(0, int(image.shape[0] / 10))
        h = image.shape[0] - h_streching
        w = image.shape[1]
    else:
        w_streching = random.randint(0, int(image.shape[1] / 10))
        h = image.shape[0]
        w = image.shape[1] - w_streching

    processed_img = tf.image.resize_images(image, [h, w], method = 1)

    return processed_img

### 色彩随机处理

def random_color_process(image, color_ordering = 0):
    if 0 == color_ordering:
        image = tf.image.random_brightness(image, max_delta = 16. / 255.)
 #       image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
  #      image = tf.image.random_hue(image, max_delta = 0.2)
   #     image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
    elif 1 == color_ordering:
#        image = tf.image.random_contrast(image, lower = 0.5, upper = 1.5)
 #       image = tf.image.random_hue(image, max_delta = 0.2)
 #       image = tf.image.random_saturation(image, lower = 0.5, upper = 1.5)
        image = tf.image.random_brightness(image, max_delta = 16. / 255.)

    # more ordering
    # 归一化
#    return tf.clip_by_value(image, 0.0, 1.0)
# 由于后续要编码 这里不能归一化
    return image


### 位置随机处理

def random_rotate(image):  
    # 随机旋转
    angle = random.uniform(-10, 10)
    image_np = np.array(image.eval())
    image_rotated = misc.imrotate(np.array(image_np),  angle, 'bicubic')
    # misc 处理输出的是nparray，转换成tensor 和前面保持一致
    image_rotated = tf.constant(image_rotated)
    return image_rotated

def random_transpose(image):
    value = random.uniform(-1, 1)
    if value > 0:
        image = tf.image.transpose_image(image)
    return image

def random_pos_process(image):
# 随机上下翻转
# 本数据集不需要上下翻转
#    image = tf.image.random_flip_up_down(image)
# 随机左右翻转
#    image = tf.image.random_flip_left_right(image)
# 随机旋转
    image = random_rotate(image)
    return image

### 尺寸随机处理

def random_size_process(image, random_crop_window):
    # 范围
    # 范围 必须大于random_crop_window
    croped_image = tf.random_crop(image, [random_crop_window[0], random_crop_window[1], 3])
    return croped_image