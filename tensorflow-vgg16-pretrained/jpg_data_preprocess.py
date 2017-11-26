import tensorflow as tf
from scipy.ndimage import imread
import os
from os.path import isfile, isdir

def fix_gpu_error():
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def _preprocess_single_pic(src_path,  dst_path):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = imread(src_path)
        resize = max(img.shape)
        img = tf.image.resize_image_with_crop_or_pad(img, resize, resize)
        img = tf.image.resize_images(img, [224, 224], method = 1)
        encoded_image = tf.image.encode_jpeg(img)
        with tf.gfile.GFile(dst_path, "wb") as f:
            f.write(encoded_image.eval())


def preprocess_pics(src_data_dir, dst_data_dir):
    fix_gpu_error()
    if not os.path.exists(dst_data_dir):
        os.makedirs(dst_data_dir)
        files = os.listdir(src_data_dir)
        for ii, file in enumerate(files, 1):
            _preprocess_single_pic(os.path.join(src_data_dir, file), os.path.join(dst_data_dir, file))
            if ii % 10 == 0 or ii == len(files):
                # 定时释放资源
                tf.reset_default_graph()
                print("{} images processed".format(ii))

