import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('0033_mp.jpg', 0)

    kernel = np.array([0, -2, -1, 6, -1, -2, 0])
    step = 4
    for i in range(img.shape[0]):
        tmp = cv2.Laplacian(img[i, :], -1, 5)
        img[i, :] = cv2.transpose(tmp)
        # for j in range(img.shape[1]):
        #     if j < step or j > img.shape[1] - step:
        #         continue
        #     img[i, j] = sum(kernel * img[i, j-step:j+step-1])

    cv2.namedWindow('LoG')
    cv2.imshow('LoG', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
