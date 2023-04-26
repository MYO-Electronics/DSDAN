import numpy as np
import cv2


def interpolation(data, bigNum):
    # only for dim is 3
    if data.shape[1] != 3:
        return None
    m = len(data)
    data_interpolation = np.zeros((m, 3, bigNum, bigNum))
    for i in range(m):
        data_interpolation[i,0] = cv2.resize(data[i,0], (bigNum, bigNum), interpolation=cv2.INTER_CUBIC)
        data_interpolation[i,1] = cv2.resize(data[i,1], (bigNum, bigNum), interpolation=cv2.INTER_CUBIC)
        data_interpolation[i,2] = cv2.resize(data[i,2], (bigNum, bigNum), interpolation=cv2.INTER_CUBIC)
    return data_interpolation
