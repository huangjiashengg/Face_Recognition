import cv2
# 加载训练文件
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
img = cv2.imread('BioID_0030.pgm')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier('D:/Users/DELL/Downloads/opencv/'
                                                     'sources/data/haarcascades/haarcascade_frontalface_default.xml')
faces = face_detector.detectMultiScale(gray)
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
    print('标签Id:', id, '置信评分：', confidence)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
