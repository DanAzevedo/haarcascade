import cv2

cap = cv2.VideoCapture('video.mp4')
classifier = cv2.CascadeClassifier('cascades/haarcascade_fullbody.xml')

while True:
    ret, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objects = classifier.detectMultiScale(imgGray, minSize=(50,50), scaleFactor=1.5)
    # print(objects)
    for x, y, w, h in objects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Captura', img)
    cv2.waitKey(10)