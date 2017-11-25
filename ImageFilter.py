# -*- coding: utf-8 -*-
import cv2
import numpy as np

print(u'正在处理中')
w_fg = 32
h_fg = 18
picflag = 3


def readpic(fn):
    # 返回图像特征码
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

# 计算余弦相似度
def get_cossimi(x,y):
    myx = np.array(x)
    myy = np.array(y)
    cos1 = np.sum(myx*myy)
    cos21 = np.sqrt(sum(myx*myx))
    cos22 = np.sqrt(sum(myy*myy))
    return cos1/float(cos21*cos22)
 
# 主程序
if __name__ == '__main__':
# 初始化图片编号
    ii=1 #初始化图片编号
    jj=2
    while True:
        fn1 ='E:\DeepLearning\PigRecog\save\pig1\p1_'+ str(ii) +'.jpeg'##读取第一张图片，请手工修改目录
        mytz= np.array(readpic(fn1))
        train_x=mytz[0].tolist()+mytz[1].tolist() + mytz[2].tolist() #计算特征值
 
        # 计算待分类图像的特征码与每个图片之间的余弦距离，余弦值越大相似度越高
        fn2 = 'E:\DeepLearning\PigRecog\save\pig1\p1_'+ str(jj) +'.jpeg'##读取第二张图片，和第一张比较，请手工修改目录
        testtz = np.array(readpic(fn2))
        simtz = testtz[0].tolist() + testtz[1].tolist() + testtz[2].tolist()
        maxtz = 0.95 #设置相似度阈值过滤图片
        nowi = 0
        nowsim = get_cossimi(train_x, simtz)
        print(nowsim)
        if nowsim<maxtz: #相似度小时保存图片，并更新基准图片
            maxtz = nowsim
            nowi = 1
            im = cv2.imread(fn2)
            path2='E:\DeepLearning\PigRecog\save\pig1\pfilter\\pf1_'+ str(jj) +'.jpeg'##请手工创建过滤图片保存目录
            cv2.imwrite(path2,im)
            ii=jj
            jj=jj+1
        else:   #相似度大时更新比较图片
            jj=jj+1
        print(u'%s 属于第 %d 类' % (fn2,nowi+1))
