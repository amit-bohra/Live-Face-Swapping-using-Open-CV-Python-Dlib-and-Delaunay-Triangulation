import cv2
import numpy as np
import dlib
from copy import deepcopy as dp




def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


img = cv2.imread('star.jpg') #provide image path here
cv2.imshow('img',img)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('68.dat')
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0,68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x,y))
    points = np.array(landmarks_points, np.int32)
    hull = cv2.convexHull(points)
    mask = np.zeros_like(img_gray)
    cv2.fillConvexPoly(mask,hull,255)
    faceimg = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(hull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles,dtype = np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret,img2 = cap.read()
else:
    ret = False
while ret:
    try:
        ret,img2 = cap.read()
        fresh = dp(img2)
        img2_new = np.zeros_like(img2)
        img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        faces2 = detector(img_gray2)
        for face in faces2:
            landmarks2 = predictor(img_gray2,face)
            landmarks_points2 = []
            for n in range(68):
                x = landmarks2.part(n).x
                y = landmarks2.part(n).y
                landmarks_points2.append((x,y))
            for triangle_index in indexes_triangles:
                tr1_pt1 = landmarks_points[triangle_index[0]]
                tr1_pt2 = landmarks_points[triangle_index[1]]
                tr1_pt3 = landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
                rect1 = cv2.boundingRect(triangle1)
                (x, y, w, h) = rect1
                cropped_triangle = img[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)
                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                  [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                  [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)
                
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
                rect2 = cv2.boundingRect(triangle2)
                (x, y, w, h) = rect2
                cropped_triangle2 = img2[y: y + h, x: x + w]
                cropped_tr2_mask = np.zeros((h, w), np.uint8)
                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                   [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                   [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
                
                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
                try:
                    cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)
                    points = np.float32(points)
                    points2 = np.float32(points2)
                    M = cv2.getAffineTransform(points, points2)
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                    triangle_area = img2_new[y: y + h, x: x + w]
                    triangle_area = cv2.add(triangle_area,warped_triangle)
                    img2_new[y: y + h, x: x + w] = triangle_area
                except:
                    pass
            points2 = np.array(landmarks_points2, np.int32)
            hull2 = cv2.convexHull(points2)
            mask2 = np.zeros_like(img_gray2)
            inv = np.bitwise_not(mask2)
            cv2.fillConvexPoly(mask2,hull2,255)
            contours,hierarchy = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            Mu = cv2.moments(cnt)
            centroidx=int(Mu['m10']/Mu['m00'])
            centroidy=int(Mu['m01']/Mu['m00'])
            img2[mask2==255]=0
            try:
                faceimg2 = cv2.seamlessClone(img2_new,img2,mask2,(centroidx,centroidy),cv2.MIXED_CLONE)
            except:
                pass
            #faceimg2 = cv2.bitwise_or(img2, img2_new)
            cv2.imshow('face2',faceimg2)
            cv2.imshow('img2',fresh)
            if cv2.waitKey(1)==27:
                ret = False
    except:
        pass
cap.release()
cv2.destroyAllWindows()
