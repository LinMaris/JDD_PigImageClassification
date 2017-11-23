# -*- coding: utf-8 -*-
import cv2
import numpy as np

print(u'���ڴ�����')
w_fg = 32
h_fg = 18
picflag = 3


def readpic(fn):
    # ����ͼ��������
    fnimg = cv2.imread(fn) #read_path
    img = cv2.resize(fnimg,(1280,720))
    w = img.shape[1]
    h = img.shape[0]
    w_interval =int( w/w_fg)
    h_interval =int( h/h_fg)
    alltz = []
    alltz.append([])
    alltz.append([])
    alltz.append([])
    for now_h in range(0,h,h_interval):
        for now_w in range(0,w,w_interval):
            b = img[now_h:now_h + h_interval, now_w:now_w+w_interval,0]
            g = img[now_h:now_h + h_interval, now_w:now_w+w_interval,1]
            r = img[now_h:now_h + h_interval, now_w:now_w+w_interval,2]
            btz = np.mean(b)
            gtz = np.mean(g)
            rtz = np.mean(r)
            alltz[0].append(btz)
            alltz[1].append(gtz)
            alltz[2].append(rtz)
    return alltz


def get_cossimi(x,y):
    myx = np.array(x)
    myy = np.array(y)
    cos1 = np.sum(myx*myy)
    cos21 = np.sqrt(sum(myx*myx))
    cos22 = np.sqrt(sum(myy*myy))
    return cos1/float(cos21*cos22)

# x��d������ʼ��

if __name__ == '__main__':
# ��ȡͼ����ȡÿ��ͼ�������
# ������������룬ͨ��ÿ�������������������������ƽ��ֵ����ȡ�������
    ii=1 #��ʼ��
    jj=2
    while True:
        fn1 ='E:\DeepLearning\PigRecog\save\pig1\p1_'+ str(ii) +'.jpeg'##���ֹ��޸�Ŀ¼
        mytz= np.array(readpic(fn1))
        train_x=mytz[0].tolist()+mytz[1].tolist() + mytz[2].tolist()

        # ���������ͼ�����������ÿ�����������֮������Ҿ��룬���������Ϊͼ����������
        fn2 = 'E:\DeepLearning\PigRecog\save\pig1\p1_'+ str(jj) +'.jpeg'##���ֹ��޸�Ŀ¼
        testtz = np.array(readpic(fn2))
        simtz = testtz[0].tolist() + testtz[1].tolist() + testtz[2].tolist()
        maxtz = 0.95 #�������ƶ���ֵ����ͼƬ
        nowi = 0
        nowsim = get_cossimi(train_x, simtz)
        print(nowsim)
        if nowsim<maxtz: #���ƶ�Сʱ����ͼƬ�������»�׼ͼƬ
            maxtz = nowsim
            nowi = 1
            im = cv2.imread(fn2)
            path2='E:\DeepLearning\PigRecog\save\pig1\pfilter\\pf1_'+ str(jj) +'.jpeg'##���ֹ���������ͼƬ����Ŀ¼
            cv2.imwrite(path2,im)
            ii=jj
            jj=jj+1
        else:   #���ƶȴ�ʱ���±Ƚ�ͼƬ
            jj=jj+1
        print(u'%s ���ڵ� %d ��' % (fn2,nowi+1))
