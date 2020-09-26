#注意：代码顺利运行需要建立以下目录
#数据集目录     ./oxbuild_images/
#中间文件存放目录       ./mid/
#测试集目录     ./aims/

import cv2
import os
import json
import numpy as np
from tqdm import tqdm
#提取的sift特征数目
sift_num=300
#k
wordCnt = 150
#kmeans终止迭代精度要求
eps=0.1
#kmeans最大迭代次数
max_iter=200
#kmeans重复次数
re_kmeans=3

#形成图像特征集
def trainSet2featureSet():
        print('提取特征集……')
        path='./oxbuild_images/'
        SIFT = cv2.xfeatures2d.SIFT_create(sift_num)
        featureSet = np.float32([]).reshape(0,128)
        img_names = os.listdir(path)
        for img_name in tqdm(img_names,ncols=100,ascii=True):
                img = cv2.imread(path+img_name,cv2.IMREAD_GRAYSCALE)
                _, des = SIFT.detectAndCompute(img, None)
                try:
                    featureSet = np.append(featureSet, des, axis=0)
                except:
                    print('\n'+img_name+':未提取到特征')
        np.save('./mid/features', featureSet)
        print('saved!')

#区块聚类
def learnVocabulary():
        print('kmeans……')
        features = np.load("./mid/features.npy")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        _, __, centers = cv2.kmeans(features, wordCnt,None, criteria, re_kmeans, cv2.KMEANS_RANDOM_CENTERS)
        np.save("./mid/vocabulary",centers)
        print('saved!')

def gaussian_kernel(x,y,sigma=0):
    kx = cv2.getGaussianKernel(y,sigma)
    ky = cv2.getGaussianKernel(x,sigma)
    return np.multiply(kx,np.transpose(ky))

#根据聚类结果生成特征向量
def feature2vector(x,y,c,dots,features,centers):
        featVec = np.zeros((1, wordCnt))
        gaussian=gaussian_kernel(x,y)
        for i in range(features.shape[0]):
                fi = features[i]
                diffMat = np.tile(fi, (wordCnt, 1)) - centers
                sqSum = (diffMat**2).sum(axis=1)
                dist = sqSum**0.5
                sortedIndices = dist.argsort()
                idx = sortedIndices[0]
                featVec[0][idx] += gaussian[int(dots[i].pt[1])][int(dots[i].pt[0])]*c
        return featVec
#对数据库内所有图片形成特征向量集
def createVectors():
        print("生成特征向量……")
        trainData = np.float32([]).reshape(0, wordCnt)
        SIFT = cv2.xfeatures2d.SIFT_create(sift_num)
        class_img_path = "./oxbuild_images/"
        centers = np.load("./mid/vocabulary.npy",allow_pickle=True)
        img_names = os.listdir(class_img_path)
        img_names.sort()
        imglist=[]
        for img_name in tqdm(img_names,ncols=100,ascii=True):
                img = cv2.imread(class_img_path+img_name,cv2.IMREAD_GRAYSCALE)
                y=img.shape[0]
                x=img.shape[1]
                c=x*y
                dots,des = SIFT.detectAndCompute(img, None)
                try:
                        featVec = feature2vector(x,y,c,dots,des,centers)
                        trainData = np.append(trainData, featVec, axis=0)
                        imglist.append(img_name)
                except:
                        img_names.remove(img_name)
                        print('\n'+img_name+'无法形成向量')
        np.save("./mid/vectors", trainData)
        with open("./mid/imglist",'w+') as f:
                f.write(json.dumps(imglist))
        print('saved!')

#读取特征向量集和图片集
def read_data():
        with open("./mid/imglist",'r') as f:
                imglist=json.loads(f.read())
        trainData=np.load('./mid/vectors.npy',allow_pickle=True)       
        return imglist,trainData

#计算两个向量的余弦相似度
def cos_sim(vector_a,vector_b):
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

#返回目标检索图像的特征向量
def aimvector(img,imgpath):
        centers = np.load("./mid/vocabulary.npy",allow_pickle=True)
        SIFT = cv2.xfeatures2d.SIFT_create(sift_num)
        y=img.shape[0]
        x=img.shape[1]
        s=x*y
        dots,des = SIFT.detectAndCompute(img, None)
        featVec = feature2vector(x,s,s,dots,des,centers)
        return dots,featVec

#对指定维度排序的sort的参数
def takeSecond(elem):
        return elem[1]

#显示图像
def takeimg(path,height):
        img=cv2.imread(path)
        rows=int(img.shape[0])
        cols=int(img.shape[1])
        n=height/rows
        size=(int(cols*n),height)
        img=cv2.resize(img,size)
        return img

#在图像上画出sift特征
def drawsift(img,dots):
        for dot in dots:
                cv2.ellipse(img,(int(dot.pt[0]),int(dot.pt[1])),(int(dot.size),int(dot.size)),dot.angle,0,360,(0,0,255),thickness=2)

#整合所有训练步骤
def train():
        trainSet2featureSet()
        learnVocabulary()
        createVectors()

if __name__ == "__main__":
        train()
        print('开始检索……')
        #每张显示前num个相似图片
        height=200
        length=1500
        x=8
        y=4
        imglist,vectors=read_data()
        #目标检索图片的储存文件夹
        aims=os.listdir('./aims/')
        for aim in aims:
                img=cv2.imread('./aims/'+aim)
                dots,aimvec=aimvector(img,'./aims/'+aim)
                #drawsift(img,dots)
                cv2.imshow(aim,img)
                cos_sims=list()
                #计算余弦相似度
                for count in range(vectors.shape[0]):
                        cos_sims.append([count,cos_sim(vectors[count],aimvec)])
                cos_sims.sort(key=takeSecond,reverse = True)
                yfirst=1
                for county in range(y):
                        xfirst=1
                        for countx in range(x):
                                path='./oxbuild_images/'+imglist[cos_sims[x*county+countx][0]]
                                if xfirst==1:
                                        temp=takeimg(path,height)
                                        xfirst=0
                                else:
                                        temp=np.hstack((temp,takeimg(path,height)))
                        rows=int(temp.shape[0])
                        cols=int(temp.shape[1])
                        n=length/cols
                        size=(length,int(rows*n))
                        temp=cv2.resize(temp,size)
                        if yfirst==1:
                                final=temp
                                yfirst=0
                        else:
                                final=np.vstack((final, temp))
                #展示结果
                cv2.imshow('result',final)
                cv2.waitKey()
                cv2.destroyAllWindows()