import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (3, 3)
frameSize = (1844, 4000)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object point and images points from all images.
objpoints = [] # 3D point in real world space
imgpoints = [] # 2D points in image plane

prev_img_shape = None

images = glob.glob('*.jpg')

for filename in images:
    image = cv2.imread(filename)
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find the chess board corners
    ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, None)

    #If found, add object point, image point (after refining them)
    if ret == True:

        objpoints.append(objectp3d)
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        #Draw and diplay corners
        cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', image)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

########################## CALIBRATION ####################################################
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

print(" Calibrated camera:")
print(ret)

print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)