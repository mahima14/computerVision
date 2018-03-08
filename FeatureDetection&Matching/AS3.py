import pandas as pd
import numpy as np
import cv2
import src.sol as sl
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def getpixelchange(frame,prev_frame):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    gx_prev = cv2.Sobel(fram_prev, cv2.CV_32F, 1, 0, ksize=1)
    gy_prev = cv2.Sobel(fram_prev, cv2.CV_32F, 0, 1, ksize=1)
    mag_prev, angle_prev = cv2.cartToPolar(gx_prev, gy_prev, angleInDegrees=True)
    frame_diff = cv2.subtract(mag_prev, mag)
    abs_sobel64f = np.absolute(frame_diff)
    sobel_8u = np.uint8(abs_sobel64f)
    cv2.imshow('Frame Difference', sobel_8u)

    return sobel_8u

def getSimilarity(edge_original, edge_now):
    gx = cv2.Sobel(edge_original, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(edge_original, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    abs_sobel64f = np.absolute(mag)
    sobel_8u = np.uint8(abs_sobel64f)
    frame_sim = cv2.absdiff(sobel_8u, edge_now)
    cv2.imshow('Frame Similarity Edge', frame_sim)

    return frame_sim

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    i = 0
    frame = list()
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    gx_prev = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    gy_prev = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag_prev, angle_prev = cv2.cartToPolar(gx_prev, gy_prev, angleInDegrees=True)
    abs_sobel64f = np.absolute(mag_prev)
    sobel_8u = np.uint8(abs_sobel64f)
    edge_original = cv2.Canny(sobel_8u, 100, 200)
    fram_prev = gray

    while (True):
        if cv2.waitKey(1) & 0xFF == ord('a'):

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edge_now = getpixelchange(gray, fram_prev)
            edge_sim = getSimilarity(edge_original, edge_now)
            ret, thresh = cv2.threshold(edge_sim, 127, 255, 0)
            _, contours, hierarchy = cv2.findContours(thresh, 1, 2)

            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            el = cv2.ellipse(edge_sim, center, (100,radius), 0,0,180,255,-1)
            cv2.imshow("Ellipse", el)

            fram_prev = gray
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()