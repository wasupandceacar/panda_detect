from PIL import Image
import PIL.ImageOps
import numpy as np
import skimage.io
import cv2
import os

# 轮廓面基阈值
AREA_THEREHOLD=500

# 弧度
ARC=0.005

# 结果筛选阈值
MATCH_THEREHOLD=0.15

def get_binimg(path):
    img = Image.open(path)
    imgs = skimage.io.imread(path)
    ttt = np.mean(imgs)
    WHITE, BLACK = 255, 0
    img = img.point(lambda x: WHITE if x > ttt else BLACK)
    img = img.convert('1')
    return img

def get_binimg_invert(path):
    img = Image.open(path)
    imgs = skimage.io.imread(path)
    ttt = np.mean(imgs)
    WHITE, BLACK = 255, 0
    img = img.point(lambda x: WHITE if x > ttt else BLACK)
    img = PIL.ImageOps.invert(img)
    img = img.convert('1')
    return img

def img_save(img, path):
    img.save(path, 'png')
    return path

def get_contours(path):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def show_contours(path):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index=0
    for cnt in contours:
        epsilon = ARC * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        imgrayc = imgray.copy()
        cv2.drawContours(imgrayc, [approx], -1, (125, 25, 0), 5)
        cv2.imshow('final'+str(index), imgrayc)
        index+=1
    cv2.waitKey(0)

def get_contour_index(path, index):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[index]
    epsilon = ARC * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return approx

def show_contour_index(path, index):
    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 150, 255, 0)
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt=contours[index]
    imgrayc = imgray.copy()
    epsilon = ARC * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(imgrayc, [approx], -1, (125, 25, 0), 5)
    cv2.imshow('final' + str(index), imgrayc)
    cv2.waitKey(0)

# 脸型对比
def filter_contours_face(contours, model):
    # 获取标准形状
    model_contour=get_contour_index(model, 1)
    # 从1开始，去除可能的外边框轮廓
    fil_results=[]
    for i in range(1, len(contours)):
        contour=contours[i]
        if cv2.contourArea(contour)>=AREA_THEREHOLD:
            #print(i, cv2.contourArea(contour), cv2.matchShapes(contour, model_contour, 1, 0.0))
            fil_results.append(cv2.matchShapes(contour, model_contour, 1, 0.0))
    return min(fil_results)

# 对比正色和反色，相加
def get_match(path):
    pngfile=path[:path.rfind(".")]+".png"
    img_save(get_binimg(path), pngfile)
    match1=filter_contours_face(get_contours(pngfile), "face.png")
    img_save(get_binimg_invert(path), pngfile)
    match2=filter_contours_face(get_contours(pngfile), "face_i.png")
    # 删除
    if os.path.exists(pngfile):
        os.remove(pngfile)
    return match1+match2

def is_panda(path):
    return (get_match(path)<=MATCH_THEREHOLD)