from scipy import ndimage
import numpy as np
import cv2

kernel = np.array([[0.0833, 0.1666, 0.0833], [0.1666, 0.000, 0.1666], [0.0833, 0.1666, 0.0833]])

def actualVal(previous, current):
    dx = cv2.Sobel(previous, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(previous, cv2.CV_32F, 0, 1, ksize=1)
    dt = current - previous
    return dx, dy, dt

##To compute Spatio Temporal Derivative
def computeDerivative(image1, image2):

    ##Kernels to calculate derivatives
	kx = np.matrix([[-1,1],[-1,1]])*.25
	ky = np.matrix([[-1,-1],[1,1]])*.25
	kt = np.ones([2,2])*.25

	dx = cv2.filter2D(image1,-1,kx) + cv2.filter2D(image2,-1,kx)
	dy = cv2.filter2D(image1,-1,ky) + cv2.filter2D(image2,-1,ky)
	dt = cv2.filter2D(image1,-1,kt) + cv2.filter2D(image2,-1,-kt)
	return (dx,dy,dt)

##To implement iterative approach of Horn Schunk optical flow estimation algorithm
def hornSchunkAlgo(previous, current, alpha, itreations):

    h, w = current.shape[:2]
    opticalFlow = np.zeros((h, w, 2), np.float32)
    dx, dy, dt = actualVal(previous, current)

    print("Value of Spatio Temporal Derivatives:",computeDerivative(previous,current))

    for i in range(itreations):
        uavg = ndimage.convolve(opticalFlow[:, :, 0], kernel, mode='constant', cval=0.0)
        vavg = ndimage.convolve(opticalFlow[:, :, 1], kernel, mode='constant', cval=0.0)
        y = alpha * alpha + np.multiply(dx, dx) + np.multiply(dy, dy)
        dyv = np.multiply(dy, vavg)
        dxu = np.multiply(dx, uavg)
        opticalFlow[:, :, 0] = uavg - (dx * (dxu + dyv + dt)) / y
        opticalFlow[:, :, 1] = vavg - (dy * (dxu + dyv + dt)) / y
    return opticalFlow

#3To draw optical flow over the image
def drawopticalflow(image, opticalFlow, steps=16):

    height, width = image.shape[:2]
    y, x = np.mgrid[steps / 2:height:steps, steps / 2:width:steps].reshape(2, -1)
    x_vector, y_vector = opticalFlow[y, x].T

    lines = np.vstack([x, y, x + x_vector, y + y_vector]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    visual = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(visual, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(visual, (x1, y1), 1, (0, 255, 0), -1)
    return visual


if __name__ == '__main__':

    cam  = cv2.VideoCapture(0)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        flow = 5 * hornSchunkAlgo(prevgray, gray, 50, 5)

        print("Optical Flow Vector:",flow)

        ch = cv2.waitKey(1) & 0xFF

        prevgray = gray
        cv2.imshow('Current Video', gray)
        if ch == ord('p') or ch == ord('P'):
            cv2.imshow('Optical Flow on a image', drawopticalflow(gray, flow))


        if ch == ord('q'):
            break


    cv2.destroyAllWindows()