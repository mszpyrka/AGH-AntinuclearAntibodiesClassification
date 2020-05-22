import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_PARAMS = {
    
    # params for detecting ROI
    'roi_window_size': 20,
    'roi_stride': 5,
    'roi_threshold': 3.,
    
    # params for extracting dataset windows
    'img_window_size': 128,
    'img_window_stride': 32,
    'img_neg_threshold': 0.05,
    'img_pos_threshold': 0.65
}


def grid_plot(arr, rows=5, cols=5, method='imshow', **kwargs):
    plots_count = rows * cols
    for plot_id, data in enumerate(arr[:plots_count]):
        plt.subplot(rows, cols, plot_id + 1)
        plt.__dict__[method](data, **kwargs)


def load_img(file, size=None, grayscale=True, top_margin=0):
    img = Image.open(file)
    
    if grayscale:
        img = img.convert('L')
    
    if size is not None:
        img = img.resize(size)
        
    return np.array(img)[top_margin:]


def to_windowed(arr, window_size, stride, flatten=True):
    arr_h, arr_w = arr.shape[:2]
    res_h = (arr_h - window_size) // stride + 1
    res_w = (arr_w - window_size) // stride + 1
    
    res = np.zeros((res_h, res_w, window_size, window_size))
    if flatten:
        res = res.reshape(res_h, res_w, -1)
        
    for h in range(res_h):
        for w in range(res_w):
            h_start = h * stride
            h_end = h * stride + window_size
            w_start = w * stride
            w_end = w * stride + window_size
            
            window = arr[h_start:h_end, w_start:w_end]
            if flatten:
                window = window.reshape(-1)
            res[h, w] = window
            
    return res


def resize_binary_mask(mask, target_size):
    mask = Image.fromarray(mask)
    mask = mask.resize(target_size)
    return np.array(mask)


def get_training_windows(img, mask, neg_threshold, pos_threshold, window_size, stride):
    windowed = to_windowed(img, window_size, stride, flatten=False)
    mask_windowed = to_windowed(mask, window_size, stride, flatten=True)
    mask_means = np.mean(mask_windowed, axis=2)
    
    neg = windowed[mask_means < neg_threshold]
    pos = windowed[mask_means > pos_threshold]
    
    return neg, pos


def img_to_dataset(img, params):
    roi_windows = to_windowed(img, window_size=params['roi_window_size'], stride=params['roi_stride'])
    roi_mask = np.std(roi_windows, axis=2) > params['roi_threshold']
    roi_mask = resize_binary_mask(roi_mask, (img.shape[1], img.shape[0]))
    
    return get_training_windows(
        img,
        roi_mask,
        window_size=params['img_window_size'],
        stride=params['img_window_stride'],
        neg_threshold=params['img_neg_threshold'],
        pos_threshold=params['img_pos_threshold']
    )
