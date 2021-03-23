"""
import cv2
im = cv2.imread('C:/Users/DELL/Desktop/image1.jpg')
# cv2.imshow('input image', im)
gray_im = cv2.cvtColor(im, code=cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_im', gray_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('C:/Users/DELL/Desktop/gray_image.jpg', gray_im)
print(gray_im.shape)

# 修改图片尺寸
gray_im1 = cv2.resize(gray_im, dsize=(360, 400))
#cv2.imshow('gray_im1', gray_im1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print(gray_im1.shape)
cv2.imwrite('C:/Users/DELL/Desktop/gray_image1.jpg', gray_im1)

# 画图
gray_im2 = cv2.resize(gray_im, dsize=(360, 700))
cv2.rectangle(gray_im2, (50, 50, 130, 130), color=(0, 255, 0), thickness=2)
cv2.imshow('resukt_im', gray_im2)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
"""
# 静态图像中的人脸识别
import cv2 as cv


def face_detect_demo(src):
    # 将图片进行灰度转换
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv.CascadeClassifier('D:/Users/DELL/Downloads/opencv/
    sources/data/haarcascades/haarcascade_frontalface_default.xml')
    # 获得人脸
    faces = face_detector.detectMultiScale(gray, 1.02, 5)
    # 获得人脸对象之后，获得其人脸的坐标定位并勾画出矩形
    for x, y, w, h in faces:
        cv.rectangle(src, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
    cv.imshow('result', src)


src = cv.imread('C:/Users/DELL/Desktop/src.jpg')
face_detect_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()"""

# 从视频中抓取人脸
import cv2 as cv
def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('D:/Users/DELL/Downloads/opencv/'
                                                     'sources/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), thickness=2, color=(0, 0, 255))
    cv.imshow('result', img)

cap = cv.VideoCapture(0)  # 其中参数可替换为视频路径如：'C:/Users/DELL/Desktop/mda-jishwjwuzgcdvfvm.mp4'
while True:
    flag, frame = cap.read()
    print('flag:', flag, ';  frame:', frame)
    if not flag:
        break
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(10):
        break

cv.destroyAllWindows()
cap.release()



















