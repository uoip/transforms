import numpy as np 
import cv2

from scipy import misc
from numpy.random import RandomState

from transforms import *
# import imageio

def test_keypoints():
    feature_detector = cv2.ORB_create(
        nfeatures=500, scaleFactor=1.2, nlevels=1, edgeThreshold=31)

    image = misc.face()    # RGB
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    print('image shape', image.shape)

    keypoints = feature_detector.detect(image[..., ::-1])
    points = [kp.pt for kp in keypoints]
    print('num of keypoints', len(keypoints))


    PRNG = RandomState()

    transform = Compose([
        [ColorJitter(prob=0.75), None],
        Expand((0.8, 1.5)),
        RandomCompose([
            RandomRotate(360),
            RandomShift(0.2)]),
        Scale(512),
        # ElasticTransform(300),
        RandomCrop(512),
        HorizontalFlip(),
        ], 
        PRNG, 
        border='constant', 
        fillval=0,
        outside_points='inf')

    results = []

    for _ in range(100):
        img, pts = transform(image, points)

        filtered = []
        for pt in pts:
            x = [abs(pt[0]), abs(pt[1])]
            if np.inf not in x and np.nan not in x:
                filtered.append(pt)

        kps = [cv2.KeyPoint(*pt, 1) for pt in filtered] 
        print('num of keypoints', len(kps))

        img = cv2.drawKeypoints(img[..., ::-1], kps, None, flags=0)
        results.append(img[..., ::-1])
        cv2.imshow('keypoints', img)
        c = cv2.waitKey(600)
        if c == 27 or c == ord('q'):   # ESC / 'q'
            break

    # imageio.mimsave('keypoints.gif', results, duration=0.5)




if __name__ == '__main__':
    test_keypoints()