import cv2


class FaceDetection:

    def __init__(self):

        self.face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.cap = cv2.VideoCapture(0)

        self.center_img = cv2.imread('image.jpeg')

    def detect(self):

        while True:
            ret, test_img = self.cap.read()  # captures frame and returns boolean value and captured image
            if not ret:
                continue
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            faces_detected = self.face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

                width_1, height_1 = y+w, x+h
                width, height = width_1-y, height_1-x
                reshape_center_img = cv2.resize(self.center_img, (height, width))

                test_img[y:width_1, x:height_1, :] = reshape_center_img[:,:,:]

            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis ', resized_img)

            if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
                break

        self.cap.release()
cv2.destroyAllWindows

dct = FaceDetection()
dct.detect()
