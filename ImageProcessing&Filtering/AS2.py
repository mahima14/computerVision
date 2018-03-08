import pandas as pd
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def convertToGrayScale(frame):
    return np.dot(frame[..., :3], [0.299, 0.587, 0.114])

def smoothing(frame):
    kernel = np.ones((5, 5), np.float32) / 25
    sframe = cv2.filter2D(frame, -1, kernel)
    return sframe

def derivativefilter(image, x, y):
    sobelx64f = cv2.Sobel(image, cv2.CV_64F, x, y, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    return sobel_8u

def normalize(image, min = 0, max = 255):
    normalized = np.zeros(image.shape)
    normalized = cv2.normalize(image,dst= normalized, alpha= min,beta= max, norm_type=cv2.NORM_MINMAX)
    return normalized

def nothing():
    pass

def mainFunction(frame, args):
    img = np.zeros((300, 512, 3), np.uint8)
    if cv2.waitKey(1) & 0xFF == ord('i'):
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        frame = cv2.resize(frame, (960, 540))
        cv2.imwrite('out.jpg', frame)

    if cv2.waitKey(1) & 0xFF == ord('g'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = cv2.resize(gray, (960, 540))
        cv2.imshow('grayFrame', im)

    if cv2.waitKey(1) & 0xFF == ord('G'):
        gray_o = convertToGrayScale(frame)
        plt.imshow(gray_o, cmap=plt.get_cmap('gray'))
        plt.xticks([]), plt.yticks([])
        plt.show()

    if cv2.waitKey(1) & 0xFF == ord('c'):

        frame[:, :, 0] = 0
        frame[:, :, 1] = 0
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow('Red Channel', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            frame[:, :, 0] = 0
            frame[:, :, 2] = 0
            frame = cv2.resize(frame, (960, 540))

            cv2.imshow('Green Channel', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                frame[:, :, 1] = 0
                frame[:, :, 2] = 0
                frame = cv2.resize(frame, (960, 540))

                cv2.imshow('Blue Channel', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sImage = cv2.GaussianBlur(gray, (5, 5), 0)
        sImage = cv2.resize(sImage, (960, 540))
        cv2.imshow('Guassian Smoothing', sImage)
        min_value = 0
        max_value = 255
        cv2.createTrackbar('Smoothing with tracker', 'Guassian Smoothing', min_value, max_value,
                           nothing)
        while (1):
            r = cv2.getTrackbarPos('Smoothing with tracker', 'Guassian Smoothing')

            sImage = cv2.GaussianBlur(gray, (5, 5), 0)
            out = cv2.inRange(sImage, r, max_value)
            out = cv2.resize(out, (960, 540))
            cv2.imshow('Guassian Smoothing', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cv2.waitKey(1) & 0xFF == ord('S'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sImage = smoothing(gray)
        sImage = cv2.resize(sImage, (960, 540))
        cv2.imshow('Smoothing with 2D convolution Filter', sImage)
        min_value = 0
        max_value = 255
        cv2.createTrackbar('Smoothing with tracker', 'Smoothing with 2D convolution Filter', min_value, max_value,
                           nothing)
        while (1):
            r = cv2.getTrackbarPos('Smoothing with tracker', 'Smoothing with 2D convolution Filter')
            sImage = smoothing(gray)
            out = cv2.inRange(sImage, r, max_value)
            out = cv2.resize(out, (960, 540))
            cv2.imshow('Smoothing with 2D convolution Filter', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cv2.waitKey(1) & 0xFF == ord('d'):
        downsample = cv2.pyrDown(frame)
        downsample = cv2.resize(downsample, (480, 270))
        cv2.imshow('Downsample Without Smoothing', downsample)

    if cv2.waitKey(1) & 0xFF == ord('D'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = smoothing(gray)
        downsample = cv2.pyrDown(dst)
        downsample = cv2.resize(downsample, (480, 270))
        cv2.imshow('Downsample With Smoothing', downsample)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagex = derivativefilter(gray, 1, 0)
        normalized = normalize(imagex)
        normalized = cv2.resize(normalized, (960, 540))
        cv2.imshow('Normalized Image- x derivative', normalized)

    if cv2.waitKey(1) & 0xFF == ord('y'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagey = derivativefilter(gray, 0, 1)
        plt.subplot(1, 3, 3), plt.imshow(imagey, cmap='gray')
        plt.title('Convolution with x derivative filter'), plt.xticks([]), plt.yticks([])
        normalized = normalize(imagey)
        normalized = cv2.resize(normalized, (960, 540))
        cv2.imshow('Normalized Image - y derivative', normalized)

    if cv2.waitKey(1) & 0xFF == ord('m'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = derivativefilter(gray, 1, 0)
        sobely = derivativefilter(gray, 1, 0)
        normalizedx = normalize(sobelx)
        normalizedy = normalize(sobely)
        mag = cv2.addWeighted(normalizedx, 0.5, normalizedy, 0.5, 0)
        mag = cv2.resize(mag, (960, 540))
        cv2.imshow('Image Gradient Magnitude', mag)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        img = np.zeros((300, 512, 3), np.uint8)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        mag = cv2.resize(mag, (960, 540))
        cv2.imshow('Gradient Vector', mag)
        cv2.createTrackbar('Pixel Controller', 'Gradient Vector', 0, 255,
                           nothing)
        while (1):
            r = cv2.getTrackbarPos('Pixel Controller', 'Gradient Vector')
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            out = cv2.inRange(mag, r, 255)
            out = cv2.resize(out, (960, 540))
            cv2.imshow('Gradient Vector', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cv2.waitKey(1) & 0xFF == ord('r'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rows, cols = gray.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        dst = cv2.warpAffine(gray, M, (cols, rows))
        dst = cv2.resize(dst, (960, 540))
        cv2.imshow('Tracker', dst)
        cv2.createTrackbar('Image Rotation', 'Tracker', 0, 360,
                           nothing)
        while (1):
            r = cv2.getTrackbarPos('Image Rotation', 'Tracker')
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), r, 1)
            dst = cv2.warpAffine(gray, M, (cols, rows))
            dst = cv2.resize(dst, (960, 540))
            cv2.imshow('Tracker', dst)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cv2.waitKey(1) & 0xFF == ord('h'):
        print("\n This program process the image and perform image manipulation using OpenCV")
        print("\n Command Line Argument:", args)
        print("\n Please see the instructions below to run this application",
                "\n Key with their function",
                "\n 'i' - reload the original image (i.e. cancel any previous processing)",
                "\n 'w' - save the current (possibly processed) image into the file 'out.jpg' ",
                "\n 'g' - convert the image to grayscale using the openCV conversion function ",
                "\n 'G' - convert the image to grayscale using your implementation of conversion function",
                "\n 'c' - cycle through the color channels of the image showing a different channel every time the key is pressed.",
                "\n 's' - convert the image to grayscale and smooth it using the openCV function. Use a track bar to control the amount of smoothing.",
                "\n	'S' - convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter. Use a track bar to control the amount of smoothing.",
                "\n 'd' - downsample the image by a factor of 2 without smoothing.",
                "\n 'D' downsample the image by a factor of 2 with smoothing.",
                "\n 'x' - convert the image to grayscale and perform convolution with an x derivative filter. Nor¬malize the obtained values to the range [0,255].",
                "\n 'y' - convert the image to grayscale and perform convolution with a y derivative filter. Normalize the obtained values to the range [0,255].",
                "\n 'm' - show the magnitude of the gradient normalized to the range [0,255]. The gradient is computed based on the x and y derivatives of the image.",
                "\n 'p' - convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K. Use a track bar to control N. Plot the vectors as short line segments of length K.",
                "\n 'r' - convert the image to grayscale and rotate it using an angle of 0 degrees. Use a track bar to control the rotation angle. The rotation of the image should be performed using an inverse map so there are no holes in it.",
                "\n 'h' - Display a short description of the program, its command line arguments, and the keys it supports.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False


if __name__ == '__main__':

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        image = cv2.imread(filename)
        imS = cv2.resize(image, (960, 540))
        cv2.imshow('frame', imS)
        while(True):
            flag = mainFunction(image, sys.argv)
            if(flag == False):
                break


    elif (len(sys.argv) < 2):
        cap = cv2.VideoCapture(0)
        while (True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            flag = mainFunction(frame, sys.argv)
            if(flag == False):
                break

    cap.release()
    cv2.destroyAllWindows()
