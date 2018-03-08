import numpy as np, cv2
from scipy import linalg
from pylab import *
from numpy import *

refPt = []
cropping = False
inputArray=[]

def clickCrop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        print(refPt)
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        print(x,y)
        cropping = False
        inputArray.append((x,y))
        cv2.imshow("LeftImage", lImage)

def getInputByMouse(lImage,rImage):

    clone1 = lImage.copy()
    clone2 = rImage.copy()

    cv2.namedWindow('LeftImage')
    cv2.setMouseCallback('LeftImage', clickCrop)
    cv2.namedWindow('RightImage')
    cv2.setMouseCallback('RightImage', clickCrop)

    while True:

        # display the image and wait for ca keypress

        cv2.imshow("LeftImage", lImage)
        cv2.imshow("RightImage", rImage)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            clone1 = clone1.copy()
            clone2 = clone2.copy()

        elif key == ord("q"):
            break
    cv2.destroyAllWindows()

def getFundamentalMatrix(a, b):

    n = a.shape[1]
    if b.shape[1] != n:
        raise ValueError("Number of points don't match.")

    A = zeros((n, 9))
    for i in range(n):
        A[i] = [a[0, i] * b[0, i], a[0, i] * b[1, i], a[0, i] * b[2, i],
                a[1, i] * b[0, i], a[1, i] * b[1, i], a[1, i] * b[2, i],
                a[2, i] * b[0, i], a[2, i] * b[1, i], a[2, i] * b[2, i]]

    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    return F / F[2, 2]


def rightEpipole(F):
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e / e[2]

def plotEpipolar(im, F,x, epipole=None, show_epipole=True):

    m, n = im.shape[:2]
    line = dot(F, x)

    t = linspace(0, n, 100)
    lt = array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    ndx = (lt >= 0) & (lt < m)
    plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = rightEpipole(F)
        plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')

    return line

def drawlinesOnImage(img1,img2,lines,points1,points2):

    c = img1.shape[1]

    for r, p1, p2 in zip(lines, points1, points2):

        x0, y0 = map(int, [0, -r[2] /r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), (255,0,0), 1)
        cv2.circle(img1, tuple(p1), 5, (255,0,0), -1)
        cv2.circle(img2, tuple(p2), 5, (255,0,0), -1)
    return img1,img2


if __name__ == '__main__':
    lImage='corridor-l.tif'
    rImage='corridor-r.tif'

    lImage=cv2.imread(lImage,0)
    rImage=cv2.imread(rImage,0)

    getInputByMouse(lImage,rImage)

    m = np.array(inputArray[:8])
    n = np.array(inputArray[8:])

    F = getFundamentalMatrix(m, n)
    right_epipole = rightEpipole(F)
    left_epipole = rightEpipole(F.T)
    print("Fundanental Matrix \n F:",F)

    print("Left Epipole \nEl:", left_epipole)

    print("Right Epipole \nEr:",right_epipole)

    getInputByMouse(lImage, rImage)
    a,b=inputArray[len(inputArray)-1]


    # Epilines corresponding to points in right image and drawing on left image
    linesl = (plotEpipolar(lImage,F,[a,b,1])).reshape(-1,3)
    imgl, img = drawlinesOnImage(lImage, rImage, linesl, m, n)

    # Epilines corresponding to points in left image and drawing on right image
    linesr = (plotEpipolar(rImage,F,[a,b,1])).reshape(-1,3)
    imgr, img = drawlinesOnImage(rImage, lImage, linesr, n, m)


    cv2.imwrite('leftImageLines.png', imgl)
    cv2.imwrite("rightImageLines.png",imgr)
