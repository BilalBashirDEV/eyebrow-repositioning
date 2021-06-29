import cv2
import numpy as np
import dlib
import imutils
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
im= cv2.imread('fc1.jpg')

im = imutils.resize(im, width=800)
#cv2.imwrite("fcc.jpg",im)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

landmarks = predictor(gray, face)


x17 = landmarks.part(17).x
y17 = landmarks.part(17).y
x18 = landmarks.part(18).x
y18 = landmarks.part(18).y
x19 = landmarks.part(19).x
y19 = landmarks.part(19).y
x20 = landmarks.part(20).x
y20 = landmarks.part(20).y
x21 = landmarks.part(21).x
y21 = landmarks.part(21).y

#import pyplot as plt
"""
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """

p1 = np.array([y17, x17])
p2 = np.array([y18, x18])
p3 = np.array([y19, x19])
p4 = np.array([y20, x20])
p5 = np.array([y21, x21])
p6=np.array([y17-18, x17])
p7=np.array([y18-18, x18])
p8=np.array([y19-18, x19])
p9=np.array([y20-18, x20])
p10=np.array([y21-18, x21])
p11=np.array([y20-18, x20-10])
p12=np.array([y21-18, x21+20])



#im=createLineIterator(p1, p2, image)
#cv2.imshow('Contours', im)
pt_a = np.array([10, 11])
pt_b = np.array([45, 67])
#points_on_line1 = np.linspace(pt_a, pt_b, 100) # 100 samples on the line
#points_on_line = np.linspace(pt_a, pt_b, np.linalg.norm(pt_a - pt_b))


#im = np.zeros((80, 80, 3), np.uint8)
for p in np.linspace(p1, p2, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)
    for o in range(12):
        f, g = p
        f = round(f)
        g = round(g)
        f-=o
        #g-=o
        # print(f,g)
        im[f, g] = im[f-25, g ]

        #b1, g1, r1 = image[x, y]

    for u in range(17):
        f, g = p
        f = round(f)
        g = round(g)
        f+=u
       # g+=u
        # print(f,g)
        im[f, g] = im[f+25, g ]
        #im[f + 25, g] = color1

for p in np.linspace(p2, p3, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)
    for o in range(10):
        f, g = p
        f = round(f)
        g = round(g)
        f-=o
        #g-=o
        # print(f,g)
        im[f, g] = im[f-35, g ]


    for u in range(17):
        f, g = p
        f = round(f)
        g = round(g)
        f+=u
        #g+=u
        # print(f,g)
        im[f, g] = im[f+25, g ]

for p in np.linspace(p3, p4, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)
    for o in range(17):
        f, g = p
        f = round(f)
        g = round(g)
        f-=o
        #g-=o
        # print(f,g)
        im[f, g] = im[f-40, g ]

    for u in range(17):
        f, g = p
        f = round(f)
        g = round(g)
        f+=u
        #g+=u
        # print(f,g)
        im[f, g] = im[f+25, g ]

for p in np.linspace(p4, p5, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)
    for o in range(12):
        f, g = p
        f = round(f)
        g = round(g)
        f-=o
        #g-=o
        # print(f,g)
        im[f, g] = im[f-30, g ]
    for u in range(17):
        f, g = p
        f = round(f)
        g = round(g)
        f+=u
        #g+=u
        # print(f,g)
        im[f, g] = im[f+25, g ]


for p in np.linspace(p6, p7, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)

    for u in range(36):
        f, h = p
        f = round(f)
        h = round(h)
        f+=u
        #g+=u
        # print(f,g)
        #im[f, g] = im[f+25, g ]
        #r, g, b = im[f + 8, g]
        #color1 = (b, g, r)
        #color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        #print(color1)
        r, g, b = im[f, h]
        e = f - 3
        d = f -1
        # v=f-3
        i=f+1
        k=f+2
        #v=f-3
        r1, g1, b1 = im[e, h]
        r2, g2, b2 = im[d, h]
        r4, g4, b4 = im[i, h]
        r5, g5, b5 = im[k, h]
        r6, g6, b6 = im[e, h+1]
        r7, g7, b7 = im[d, h+2]
        r8, g8, b8 = im[i, h+3]
        r9, g9, b9 = im[k, h+4]
        r3 = round((int(r)+int(r1)+int(r2)+int(r4)+int(r5)+int(r6)+int(r7)+int(r8)+int(r9))/9)
        g3 = round((int(g)+int(g1)+int(g2)+int(g4)+int(g5)+int(g6)+int(g7)+int(g8)+int(g9))/9)
        b3 = round((int(b)+int(b1)+int(b2)+int(b4)+int(b5)+int(b6)+int(b7)+int(b8)+int(b9))/9)

        # color1 = (b, g, r)
        # color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        color3 = (r3, g3, b3)
        color3 = (int(color3[0]), int(color3[1]), int(color3[2]))
        im[f, h] = color3
        #im[f-3:f+3, h:h+12] = cv2.GaussianBlur(im[f-3:f+3, h:h+12], (3,3),cv2.BORDER_DEFAULT)
        # print(color1)
for p in np.linspace(p7, p8, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)

    for u in range(36):
        f, h = p
        f = round(f)
        h = round(h)
        f+=u
        #g+=u
        # print(f,g)
        #im[f, g] = im[f+25, g ]
        #r, g, b = im[f + 8, g]
        #color1 = (b, g, r)
        #color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        #print(color1)
        r, g, b = im[f, h]
        e = f - 3
        d = f - 1
        # v=f-3
        i = f + 1
        k = f + 2
        #v=f-3
        r1, g1, b1 = im[e, h]
        r2, g2, b2 = im[d, h]
        r4, g4, b4 = im[i, h]
        r5, g5, b5 = im[k, h]
        r6, g6, b6 = im[e, h+1]
        r7, g7, b7 = im[d, h+2]
        r8, g8, b8 = im[i, h+3]
        r9, g9, b9 = im[k, h+4]
        r3 = round((int(r)+int(r1)+int(r2)+int(r4)+int(r5)+int(r6)+int(r7)+int(r8)+int(r9))/9)
        g3 = round((int(g)+int(g1)+int(g2)+int(g4)+int(g5)+int(g6)+int(g7)+int(g8)+int(g9))/9)
        b3 = round((int(b)+int(b1)+int(b2)+int(b4)+int(b5)+int(b6)+int(b7)+int(b8)+int(b9))/9)


        # color1 = (b, g, r)
        # color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        color3 = (r3, g3, b3)
        color3 = (int(color3[0]), int(color3[1]), int(color3[2]))
        im[f, h] = color3
        #im[f - 3:f + 3, h:h + 12] = cv2.GaussianBlur(im[f - 3:f + 3, h:h + 12], (5, 5), cv2.BORDER_DEFAULT)
        # print(color1)
for p in np.linspace(p8, p9, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)

    for u in range(36):
        f, h = p
        f = round(f)
        h = round(h)
        f+=u
        #g+=u
        # print(f,g)
        #im[f, g] = im[f+25, g ]
        #r, g, b = im[f + 8, g]
        #color1 = (b, g, r)
        #color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        #print(color1)
        r, g, b = im[f, h]
        e = f - 3
        d = f - 1
        # v=f-3
        i = f + 1
        k = f + 2
        #v=f-3
        r1, g1, b1 = im[e, h]
        r2, g2, b2 = im[d, h]
        r4, g4, b4 = im[i, h]
        r5, g5, b5 = im[k, h]
        r6, g6, b6 = im[e, h+1]
        r7, g7, b7 = im[d, h+2]
        r8, g8, b8 = im[i, h+3]
        r9, g9, b9 = im[k, h+4]
        r3 = round((int(r)+int(r1)+int(r2)+int(r4)+int(r5)+int(r6)+int(r7)+int(r8)+int(r9))/9)
        g3 = round((int(g)+int(g1)+int(g2)+int(g4)+int(g5)+int(g6)+int(g7)+int(g8)+int(g9))/9)
        b3 = round((int(b)+int(b1)+int(b2)+int(b4)+int(b5)+int(b6)+int(b7)+int(b8)+int(b9))/9)


        # color1 = (b, g, r)
        # color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        color3 = (r3, g3, b3)
        color3 = (int(color3[0]), int(color3[1]), int(color3[2]))
        im[f, h] = color3
        #im[f - 3:f + 3, h:h + 12] = cv2.GaussianBlur(im[f - 3:f + 3, h:h + 12], (3, 3), cv2.BORDER_DEFAULT)
        # print(color1)

for p in np.linspace(p9, p10, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)

    for u in range(36):
        f, h = p
        f = round(f)
        h = round(h)
        f += u
        #h+=u
        # print(f,g)
        #im[f, g] = im[f+25, g ]
        #r, g, b = im[f + 8, g]
        #color1 = (b, g, r)
        #color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        #print(color1)
        r, g, b = im[f, h]
        e = f -3
        d = f -1
        # v=f-3
        i=f+1
        k=f+3

        #v=f-3
        r1, g1, b1 = im[e, h]
        r2, g2, b2 = im[d, h]
        r4, g4, b4 = im[i, h]
        r5, g5, b5 = im[k, h]
        r6, g6, b6 = im[e, h+1]
        r7, g7, b7 = im[d, h+2]
        r8, g8, b8 = im[i, h+3]
        r9, g9, b9 = im[k, h+4]
        r3 = round((int(r)+int(r1)+int(r2)+int(r4)+int(r5)+int(r6)+int(r7)+int(r8)+int(r9))/9)
        g3 = round((int(g)+int(g1)+int(g2)+int(g4)+int(g5)+int(g6)+int(g7)+int(g8)+int(g9))/9)
        b3 = round((int(b)+int(b1)+int(b2)+int(b4)+int(b5)+int(b6)+int(b7)+int(b8)+int(b9))/9)


        # color1 = (b, g, r)
        # color1 = (int(color1[2]), int(color1[1]), int(color1[0]))
        color3 = (r3, g3, b3)
        color3 = (int(color3[0]), int(color3[1]), int(color3[2]))
        im[f, h] = color3
        #im[f - 3:f + 5, h:h + 24] = cv2.GaussianBlur(im[f - 3:f + 5, h:h + 24], (3,3), cv2.BORDER_DEFAULT)
        # print(color1)
'''
for p in np.linspace(p11, p12, 100):
    #cv2.circle(im, tuple(np.int32(p)), 1, (255,0,0), -1)
    e=0
    for u in range(37):

        f, g = p
        f = round(f)
        g = round(g)
        i=f
        #g+=u
        if u*2 !=0:
            i+=round((u+e))
           # u += 1
        # print(f,g)
            im[f, g] = im[i, g+10 ]
            #u+=6
            e+=2*e

'''

cv2.imwrite('true.jpg',im)
cv2.imshow('l',im)
cv2.waitKey(0)