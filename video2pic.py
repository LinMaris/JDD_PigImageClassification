import cv2
# import numpy as np
import os

def video2Img(read_path,save_path,each_video_name):

    # 创建视频对处理对象
    cap = cv2.VideoCapture(read_path)

    # 设置视频处理的起始帧数，默认为1
    img_point = 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, img_point)
    # 视频总帧数
    img_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not cap.isOpened() :
        print('视频读取错误')
        return

    #  cap.isOpened()  判断视频是否打开
    while (img_point <= img_total):

        print(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 当前视频显示帧数，默认累加，一帧帧
        ret, frame = cap.read()  # 视频读取

        # cv2.imshow("video", frame)  # 图片展示
        cv2.imwrite(save_path + each_video_name + '_%d.jpeg' % (img_point), frame)  # 图片保存

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_point += 1
    cap.release()     # 如果因为视频过长，导致内存占用过高，可以采取分批次处理，这里不做展开
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # 鉴于数据集过大,数据存储在硬盘指定目录
    videos_src_path = r'C:\Users\hasee\Downloads\百度云下载\JDD_PigIdentify\train'
    videos_save_path = r'C:\Users\hasee\Downloads\BaiduYun_Download\5_Pig_Img\save'
    # 保存地址不能有中文

    videos = os.listdir(videos_src_path)
    videos = filter(lambda x: x.endswith('mp4'), videos)

    for each_video in videos:
        # 获取每个视频的名字，并以名字创建文件夹保存
        each_video_name, _ = each_video.split('.')
        if not os.path.isdir(videos_save_path):
            os.mkdir(videos_save_path)
        if not os.path.isdir(videos_save_path + '/' + each_video_name):
            os.mkdir(videos_save_path + '/' + each_video_name)

        # 每个视频的保存地址和源地址
        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '\\'
        each_video_read_full_path = os.path.join(videos_src_path, each_video)

        # video提取
        video2Img(each_video_read_full_path, each_video_save_full_path, each_video_name)