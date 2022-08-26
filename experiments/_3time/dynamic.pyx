# Refer to https://github.com/kr7/DCNN/blob/main/DCNN.ipynb

import numpy as np
cimport numpy as np
from tqdm import tqdm

def dtw(np.ndarray[np.float_t, ndim = 1] ts1,
    np.ndarray[np.float_t, ndim = 1] ts2):

    cdef int LEN_TS1
    cdef int LEN_TS2
    cdef int i
    cdef int j
    cdef np.ndarray[np.float_t, ndim = 2] dtw_matrix

    LEN_TS1 = len(ts1)
    LEN_TS2 = len(ts2)

    dtw_matrix = np.zeros((LEN_TS1, LEN_TS2), dtype=np.float)

    dtw_matrix[0, 0] = abs(ts1[0] - ts2[0])

    for i in range(1, LEN_TS1):
        dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + abs(ts1[i] - ts2[0])

    for j in range(1, LEN_TS2):
        dtw_matrix[0, j] = dtw_matrix[0, j - 1] + abs(ts1[0] - ts2[j])

    for i in range(1, LEN_TS1):
        for j in range(1, LEN_TS2):
            dtw_matrix[i, j] = min(dtw_matrix[i - 1, j - 1], dtw_matrix[i - 1, j],
                               dtw_matrix[i, j - 1]) + abs(ts1[i] - ts2[j])

    return dtw_matrix[len(ts1) - 1, len(ts2) - 1]


def dc_activations(data, convolutional_filters):
    num_instances = len(data)
    length_of_time_series = len(data[0])
    num_conv_filters = len(convolutional_filters)
    conv_filter_size = len(convolutional_filters[0][0])

    activations = np.zeros((num_instances, num_conv_filters,
                            length_of_time_series - conv_filter_size + 1))
    for i in tqdm(range(num_instances), total=num_instances):
        for j in range(length_of_time_series - conv_filter_size + 1):
            for k in range(num_conv_filters):
                activations[i, k, j] = dtw(convolutional_filters[k][0],
                                           data[i, j:j + conv_filter_size])

    return activations
