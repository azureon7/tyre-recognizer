# import packages

import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


def cif_step1(img):
    img_small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 10)
    h = img_small.shape[0]
    # identifico circonferenze
    circles = cv2.HoughCircles(edges, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, circles=None,
                               minRadius=int(h / 4), maxRadius=int(h / 2.2), param1=100, param2=175)
    circles = np.int0(np.array(circles))
    circles_df = pd.DataFrame(circles[0], columns=['X', 'Y', 'R'])
    del circles
    X = int(circles_df.X.mean())  # centro medio, coordinata X
    Y = int(circles_df.Y.mean())  # centro medio, coordinata Y
    Rmax = int(circles_df.R.max())  # raggio massimo
    del circles_df
    X1 = int(X * 3 / 0.25)
    Y1 = int(Y * 3 / 0.25)
    R1 = int(Rmax * 3 / 0.25) + 10
    img_big = cv2.resize(img, (0, 0), fx=3, fy=3)
    img_crop = img_big[Y1 - R1:Y1 + R1, X1 - R1:X1 + R1].copy()
    del img_big
    img_polar = cv2.logPolar(img_crop, (R1, R1), 72.5 * 3 / 0.25, cv2.WARP_FILL_OUTLIERS)
    del img_crop
    img_rotated = cv2.rotate(img_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    del img_polar
    img_rotated = img_rotated[0:int(img_rotated.shape[0] * 0.2), :]
    return img_rotated


def cif_step2(img_rotated):
    altezza = img_rotated.shape[0]
    larghezza = img_rotated.shape[1]
    for i in range(altezza):
        #print(i, larghezza/5)
        help_img = img_rotated[i:i+1, :int(larghezza/5)]
        pixels = help_img.reshape(-1, 3)
        counts = defaultdict(int)
        for pixel in pixels:
            if sum(pixel) == 0:
                counts['blacks'] += 1
        if counts['blacks'] < 50:
            H = i
            break
    output = img_rotated[H:, :]
    return output


def cif_step3(img_rotated):
    subimg = cv2.cvtColor(img_rotated[50:, :100], cv2.COLOR_BGR2GRAY)
    H = []
    for i in range(100):
        subsubimg = subimg[:, i:i + 1].reshape(-1, 1)
        valori = []
        k = 0
        for pix in subsubimg:
            valori.append(pix)
            k += 1
            if k > 50 and abs(pix - np.median(valori)) > 100:
                H.append(k + 50)
                break
    v = int(np.median(H) + 20)
    output = img_rotated[:v, :]
    return output


def cif_isoutlier(img):
    # altezza
    h = img.shape[0]
    # colore + frequente
    Z = np.float32(img[:, :200].reshape((-1, 3)))
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = map(tuple, center[label.flatten()])
    # salvo rgb e controllo tipo boxplot se sono outliers
    r = Counter(res).most_common(1)[0][0][0]
    g = Counter(res).most_common(1)[0][0][1]
    b = Counter(res).most_common(1)[0][0][2]
    X = pd.Series([h, r, g, b])
    Q1 = pd.Series([271.5, 59.0, 59.0, 59.0])
    Q3 = pd.Series([357.0, 88.0, 86.0, 86.0])
    IQR = Q3 - Q1
    isoutlier = list((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))
    cond1 = isoutlier[0]
    cond2 = isoutlier[1] | isoutlier[2] | isoutlier[3]
    outlier = (cond1 or cond2)
    return outlier
