# import packages

import cv2
import numpy as np
import pandas as pd

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def cif_step1(img):
    img = image_resize(img, width=6000)
    img_small = image_resize(img, width=320)
    const1 = 320.0 / 6000
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 20)
    h = img_small.shape[0]
    # identifico circonferenze
    # step 1
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, circles=None,
                               minRadius=int(h / 3.5), maxRadius=int(h / 2.1), param1=50, param2=50)
    circles = np.int0(np.array(circles))
    circles_df = pd.DataFrame(circles[0], columns=['X', 'Y', 'R'])
    Xmedian = int(circles_df.X.median())  # centro medio, coordinata X
    Ymedian = int(circles_df.Y.median())  # centro medio, coordinata Y
    circles_df['distance']=((circles_df.X-Xmedian)**2+(circles_df.Y-Ymedian)**2)**0.5
    circles_df = circles_df[circles_df.distance < 10]
    Rmax = int((2.5+circles_df.R.max())/const1)
    X = int(circles_df.iloc[:min(50, len(circles_df)), 0].mean()/const1)
    Y = int(circles_df.iloc[:min(50, len(circles_df)), 1].mean()/const1)
    img = img[Y-Rmax:Y+Rmax,X-Rmax:X+Rmax]
    img = cv2.resize(img, (5000, 5000))
    return img


def cif_step2_1(img):
    img_polar = cv2.logPolar(img, (2500, 2500), 635, cv2.WARP_FILL_OUTLIERS)
    img_rotated = cv2.rotate(img_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rotated = img_rotated[30:360, :]
    return img_rotated


def cif_step2_2(img):
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_polar = cv2.logPolar(img, (2500, 2500), 635, cv2.WARP_FILL_OUTLIERS)
    img_rotated = cv2.rotate(img_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_rotated = img_rotated[30:360, :]
    return img_rotated
