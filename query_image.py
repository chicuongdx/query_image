#Bài tập thực hành - Lập trình máy học cho python - CS116.M11

#MSSV: 19521299
#Tên: Nguyễn Chí Cường
#Ngày: 12/1/2021
#Tuần 13


from keras.applications.vgg16 import VGG16
import os
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Bước 1: Load pretrained model VGG16
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

model = tf.keras.Sequential(
    [
        vgg,
        tf.keras.layers.Flatten()
    ]
)

#Bước 2: Load các ảnh trong 1 thư mục
f = 'image/'
lst_dir = [f + name for name in os.listdir(f)]
orig_image = [cv2.imread(x) for x in lst_dir]
images = np.array([cv2.resize(cv2.imread(x), (224, 224)) / 255.0 for x in lst_dir])
#Bước 3: Rút trích đặc trưng các ảnh trong thư mục
feature_images = model.predict(images)

def find_nearestImg(path_image='', k_top=3): #with folder image
    #Bước 4: đọc ảnh truy vấn, Bước 5: Rút trích đặc trưng ảnh truy vấn
    orig = cv2.imread(path_image)
    img = np.array([cv2.resize(orig, (224, 224)) / 255.0])
    feature = model.predict(img)[0].reshape(1, -1)

    #load NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k_top).fit(feature_images)

    #Bước 6: Dùng mô hình KNN search với top K = 3 để tìm các đặc trưng gần với đặc trưng truy vấn nhất
    nearest_idx = nbrs.kneighbors(feature.reshape(1, -1), return_distance=False)
    nearest_img = [orig_image[x] for x in nearest_idx[0]]
    return orig, nearest_img

#Bước 7:show top K ảnh gần nhất đó
def plot_nearest(path_image=''):
    plt.figure(figsize=(20,20))
    image, nearest = find_nearestImg(path_image)
    plt.subplot(1, 4, 1)
    plt.title('Ảnh truy vấn')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    idx = 1
    for img in nearest:
        plt.subplot(1, 4, idx + 1)
        plt.title('Ảnh gần giống nhất thứ ' + str(idx))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        idx += 1

    plt.show()

def plot_MulNearest(folder=''):
    plt.figure(figsize=(30, 30))

    exam_path = [folder + name for name in os.listdir(folder)]
    idx = 1
    for path in exam_path:
        image, nearest = find_nearestImg(path)
        plt.subplot(6, 4, idx)
        plt.title('Ảnh truy vấn')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        
        jdx = 1
        for img in nearest:
            plt.subplot(6, 4, idx+1)
            plt.title('Ảnh gần giống nhất thứ ' + str(jdx))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            jdx += 1
            idx += 1
        idx += 1
    plt.show()

##################################################### main:
f_exam = 'example/'
exam_path = [f_exam + name for name in os.listdir(f_exam)]

plot_nearest(exam_path[0])