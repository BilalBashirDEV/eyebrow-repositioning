import cv2
import numpy as np
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def get_inpainted(img):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gs, 135, 1, cv2.THRESH_BINARY)
    mask_features = 1 - thresh #1 for the dark pixels, 0 everywhere else
    #enlarge the mask a little bit...
    mask_features = cv2.dilate(mask_features, np.ones((8, 8), np.uint8))
    return cv2.inpaint(img, mask_features, 8, cv2.INPAINT_NS)



img_source = cv2.imread("fff.jpg")
img_source = cv2.resize(img_source,(590,535))
img_dest = img_source[:]

gray = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

landmarks = predictor(gray, face)


i = landmarks.part(17).x
u = landmarks.part(17).y

e = landmarks.part(21).x
r = landmarks.part(21).y

#sp=(i,u-2)
#ep=(e,r-8)
print(u)
print(e)
print(r)
print(i)

t = landmarks.part(22).x
w = landmarks.part(22).y

l = landmarks.part(26).x
k = landmarks.part(26).y

sp1=(t,w)
ep1=(l,k-14)

print(sp1)
print(ep1)
#manually select two areas of interest
img_dest[r-21:u+6, i-4:e+4] = get_inpainted(img_source[r-21:u+6, i-4:e+4]) #eyebrows to nostrils
img_dest[k-32:w+5,t:l] = get_inpainted(img_source[k-32:w+5,t:l]) #mouth
img_dest[r - 35:u+9, i-9:e +9] = cv2.GaussianBlur(img_dest[r - 35:u+9, i-9:e +9],(5,5),cv2.BORDER_DEFAULT)
img_dest[k - 31:w + 20, t-29:l +29] = cv2.GaussianBlur(img_dest[k-31:w+20,t-29:l+29],(5,5),cv2.BORDER_DEFAULT)
cv2.imwrite("fcc1.jpg", img_dest)