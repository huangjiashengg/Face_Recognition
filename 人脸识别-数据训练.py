import os
import cv2
from PIL import Image
import numpy as np
def getImageAndLabels(path):
    facesSmaple = []
    ids = []
    face_detector = cv2.CascadeClassifier('D:/Users/DELL/Downloads/opencv/'
                                         'sources/data/haarcascades/haarcascade_frontalface_default.xml')
    ImagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # 遍历列表中的图片
    for imagepath in ImagePaths:
        PLT_imag = Image.open(imagepath).convert('L')
        img_numpy = np.array(PLT_imag, 'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        # 获取每张图片的id
        id = int(os.path.split(imagepath)[1][8:10])
        for x, y, w, h in faces:
            facesSmaple.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    print(ids)
    return facesSmaple, ids

if __name__ == '__main__':
    # 图片路劲
    path = './data/jm/'
    # 获取图像数组和Id标签数组
    faces, ids = getImageAndLabels(path)
    # 循环获取对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    # 保存文件
    recognizer.write('trainer/trainer.yml')
