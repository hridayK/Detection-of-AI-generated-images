import numpy as np
import cv2
from scipy.ndimage import rotate

def apply_filter_a(src:np.ndarray):
    src_copy = np.copy(src)
    base_filter = [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -1,  0,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]
    f1 = np.asarray([
        rotate(input=base_filter,angle=45,reshape=False),
        base_filter,
        rotate(input=base_filter,angle=-45,reshape=False),
        rotate(input=base_filter,angle=-90,reshape=False),
        rotate(input=base_filter,angle=-135,reshape=False),
        rotate(input=base_filter,angle=180,reshape=False),
        rotate(input=base_filter,angle=135,reshape=False),
        rotate(input=base_filter,angle=90,reshape=False),
    ])
    
    img = cv2.filter2D(src=src_copy, kernel=f1[0], ddepth=-1)
    for filter in f1[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img

def apply_filter_b(src:np.ndarray):
    src_copy = np.copy(src)
    base_filter = [[ 0,  0, -1,  0,  0],
                    [ 0,  0,  3,  0,  0],
                    [ 0,  0, -3,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]]
    # removed the noise adding kernel
    # f2 = np.asarray([
    #     rotate(input=base_filter,angle=45,reshape=False),
    #     base_filter,
    #     rotate(input=base_filter,angle=-45,reshape=False),
    #     rotate(input=base_filter,angle=-90,reshape=False),
    #     rotate(input=base_filter,angle=-135,reshape=False),
    #     rotate(input=base_filter,angle=180,reshape=False),
    #     rotate(input=base_filter,angle=135,reshape=False),
    #     rotate(input=base_filter,angle=90,reshape=False)
    # ])
    f2 = np.asarray([
        base_filter,
        rotate(input=base_filter,angle=-90,reshape=False),
        rotate(input=base_filter,angle=180,reshape=False),
        rotate(input=base_filter,angle=90,reshape=False)
    ])
                    
    
    img = cv2.filter2D(src=src_copy, kernel=f2[0], ddepth=-1)
    for filter in f2[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img



def apply_filter_c(src:np.ndarray):
    src_copy=np.copy(src)
    base_filter = [[ 0,  0,  0,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0, -2,  0,  0],
                    [ 0,  0,  1,  0,  0],
                    [ 0,  0,  0,  0,  0]]
    f3 = np.asarray([
        base_filter,
        rotate(input=base_filter,angle=-90,reshape=False),
        rotate(input=base_filter,angle=45,reshape=False),
        rotate(input=base_filter,angle=-45,reshape=False)
    ])
    
    img = cv2.filter2D(src=src_copy, kernel=f3[0], ddepth=-1)
    for filter in f3[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img


def apply_filter_d(src:np.ndarray):
    src_copy=np.copy(src)
    base_filter = [[ 0,  0,  0,  0,  0],
                    [ 0, -1,  2, -1,  0],
                    [ 0,  2, -4,  2,  0],
                    [ 0,  0,  0,  0,  0],
                    [ 0,  0,  0,  0,  0]]
    f4 = np.asarray([
        base_filter,
        rotate(input=base_filter,angle=90,reshape=False),
        rotate(input=base_filter,angle=180,reshape=False),
        rotate(input=base_filter,angle=-90,reshape=False)
    ])
    
    img = cv2.filter2D(src=src_copy, kernel=f4[0], ddepth=-1)
    for filter in f4[1:]:
        img = cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img

def apply_filter_e(src:np.ndarray):
    src_copy=np.copy(src)
    base_filter = [[  1,   2,  -2,   2,   1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  0,   0,   0,   0,   0],
                    [  0,   0,   0,   0,   0]]
    
    f5 = np.asarray([
        base_filter,
        rotate(input=base_filter,angle=90,reshape=False),
        rotate(input=base_filter,angle=180,reshape=False),
        rotate(input=base_filter,angle=-90,reshape=False)
    ])
    
    img = cv2.filter2D(src=src_copy, kernel=f5[0], ddepth=-1)
    for filter in f5[1:]:
        img=cv2.add(img,cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))

    return img

def apply_filter_f(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.asarray([[ 0,  0,  0,  0,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  2,  -4,  2,  0],
                    [ 0,  -1,  2, -1,  0],
                    [ 0,  0,  0,  0,  0]])
    
    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img


def apply_filter_g(src:np.ndarray):
    src_copy=np.copy(src)
    f5 = np.asarray([[ -1,   2,  -2,   2,  -1],
                    [  2,  -6,   8,  -6,   2],
                    [ -2,   8, -12,   8,  -2],
                    [  2,  -6,   8,  -6,   2],
                    [ -1,   2,  -2,   2,  -1]])
    
    img = cv2.filter2D(src=src_copy, kernel=f5, ddepth=-1)
    return img

def apply_all_filters(src:np.ndarray):
    src_copy = np.copy(src)
    # return apply_filter_a(src_copy) + apply_filter_b(src_copy) + apply_filter_c(src_copy) + \
    # apply_filter_d(src_copy) + apply_filter_e(src_copy) + apply_filter_f(src_copy)
    return apply_filter_a(src_copy) + apply_filter_b(src_copy) + apply_filter_c(src_copy) + \
    apply_filter_d(src_copy) + apply_filter_f(src_copy)