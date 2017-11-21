import cv2
import numpy as np
import os
import tensorflow as tf

rootDir = './save/1/'
def makeData():
    train_writer = tf.python_io.TFRecordWriter('train.tfrecords')
    valid_writer = tf.python_io.TFRecordWriter('valid.tfrecords')

    # test_writer = tf.python_io.TFRecordWriter('test.tfrecords')

    img_size = (256,256)   #待定
    for parent, dirnames,filenames in os.walk(rootDir):
        for filename in filenames:
            if filename.endswith('.jpeg'):
                label = np.zeros(shape=[30],dtype=np.uint8)
                each_video_path = os.path.join(parent,filename)
                img = cv2.imread(each_video_path)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                result = filename.split('_')[0]
                label[int(result) - 1] = 1     # one-hot
                # print(label)
                img_raw = img.tostring() # 将图像数据转化为包含像素数据的字符串
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
                            'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        }
                    )
                )
                rn = int(filename.split('_')[1][:-5])
                if rn % 5 == 0:
                    valid_writer.write(example.SerializeToString())
                else:
                    train_writer.write(example.SerializeToString())
    train_writer.close()
    valid_writer.close()

def readRecords(filename):
    # 生成文件名队列,此处默认为当前创建的文件
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'data_raw': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string),
                                       }
                                       )

    img = tf.decode_raw(features['data_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.decode_raw(features['label'], tf.uint8)

    label = tf.reshape(label, [30])

    return img, label

def test_records():
    img, label = readRecords("train.tfrecords")

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        threads = tf.train.start_queue_runners(sess=sess)

        # 输出10个样本
        for i in range(10):
            val, l = sess.run([img_batch, label_batch])
            # l = to_categorical(l, 12)
            print(val.shape, l)


if __name__ == '__main__':
    # makeData()

    test_records()
