import os
import nibabel as nib
import random
import cv2
from scipy.ndimage import binary_dilation
import numpy as np

fold0_validation_path = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_results/Dataset086_TCIA/nnUNetTrainerFocalLoss__nnUNetPlans__3d_fullres/fold_0/validation"
fold1_validation_path = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_results/Dataset086_TCIA/nnUNetTrainerFocalLoss__nnUNetPlans__3d_fullres/fold_1/validation"
fold2_validation_path = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_results/Dataset086_TCIA/nnUNetTrainerFocalLoss__nnUNetPlans__3d_fullres/fold_2/validation"
fold3_validation_path = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_results/Dataset086_TCIA/nnUNetTrainerFocalLoss__nnUNetPlans__3d_fullres/fold_3/validation"
fold4_validation_path = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_results/Dataset086_TCIA/nnUNetTrainerFocalLoss__nnUNetPlans__3d_fullres/fold_4/validation"

def proper_prediction(patient):
    if os.path.exists(fold0_validation_path+ '/'+ patient):
        return nib.load(os.path.join(fold0_validation_path, patient)).get_fdata()
    
    elif os.path.exists(fold1_validation_path+ '/'+ patient):
        return nib.load(os.path.join(fold1_validation_path, patient)).get_fdata()
    
    elif os.path.exists(fold2_validation_path+ '/'+ patient):
        return nib.load(os.path.join(fold2_validation_path, patient)).get_fdata()
    
    elif os.path.exists(fold3_validation_path+ '/'+ patient):
        return nib.load(os.path.join(fold3_validation_path, patient)).get_fdata()
    
    else:
        return nib.load(os.path.join(fold4_validation_path, patient)).get_fdata()
    
def contrast_stretching(img):
    minmax_img = np.zeros_like(img)
    for i in range(img.shape[2]):
        minmax_img[:, :, i] = cv2.normalize(img[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
    return minmax_img.astype(np.uint8)

def random_dilation(binary_image, min_size=1, max_size=5):
    # Generate random size for structuring element
    size_x = random.randint(min_size, max_size)
    size_y = random.randint(min_size, max_size)
    size_z = random.randint(min_size, max_size)

    # Create the random-sized structuring element
    structuring_element = np.ones((size_x, size_y, size_z), dtype=bool)

    # Perform dilation
    return binary_dilation(binary_image, structure=structuring_element)

def export_contours_to_onnx_model(pred):
    three_channel_pred = np.repeat(pred[:, :, None], 3, axis=2)
    gray = cv2.cvtColor(three_channel_pred.astype('uint8'), cv2.COLOR_BGR2GRAY)
    # threshold
    thresh = cv2.threshold(gray, 0.5, 1, cv2.THRESH_BINARY)[1]

    # get contours
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    
    onnx_box_labels = np.array([2,3])  # 2 is a top-left box corner, 3 is a bottom-right box corner
    onnx_box_labels_copied = onnx_box_labels.copy() # For adding the correct labels later on in the loop

    # if contours:
    if len(contours) == 1:                         # Splitting based on having one or more than one contours is because of the concatenation process
        y,x,h,w = cv2.boundingRect(contours[0])
        onnx_box_coords = np.array([[y, x], [y+h, x+w]])
        onnx_coord = onnx_box_coords[None, :, :]
        onnx_label = onnx_box_labels[None, :].astype(np.float32)
        return onnx_coord, onnx_label
    else:
        y,x,h,w = cv2.boundingRect(contours[0])
        onnx_box_coords = np.array([[y, x], [y+h, x+w]])
        for cntr in range(1, len(contours)):
            y,x,h,w = cv2.boundingRect(contours[cntr])
            onnx_coords = np.array([[y, x], [y+h, x+w]])
            onnx_box_coords = np.append(onnx_box_coords, onnx_coords, axis=0)
            onnx_box_labels = np.append(onnx_box_labels, onnx_box_labels_copied, axis=0)
        onnx_box_coords = onnx_box_coords[None, :, :]
        onnx_box_labels = onnx_box_labels[None, :].astype(np.float32)
        return onnx_box_coords, onnx_box_labels
    # else:  ## Prediction slice is empty
    #     input_point = np.array([[0.0, 0.0]])
    #     input_label = np.array([-1])
    #     onnx_box_coords = input_point[None, :, :]
    #     onnx_box_labels = input_label[None, :].astype(np.float32)
    #     return onnx_box_coords, onnx_box_labels

def preprocess_CT(image):
    WINDOW_LEVEL = 40  
    WINDOW_WIDTH = 400 
    lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
    upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
    image_preprocessed = np.clip(image, lower_bound, upper_bound)
    image_preprocessed = (
        (image_preprocessed - np.min(image_preprocessed))
        / (np.max(image_preprocessed) - np.min(image_preprocessed))
        * 255.0
    )
    return np.uint8(image_preprocessed)

class study:
    def __init__(self, study_name):
        self.study_name = study_name
        self.dice       = None
        self.recall     = None
        self.precision  = None
        self.specifity  = None
        self.FNR        = None
    def compute_dice(self, arr1, arr2):
        overlap = np.multiply(arr1, arr2)
        
        tp = np.count_nonzero(overlap)      # arr1 ground truth
        fn = np.count_nonzero(arr1-overlap) # arr2 prediction
        fp = np.count_nonzero(arr2-overlap)
        
        if 2*tp+fp+fn == 0:
            return np.nan
        else:
            return 2*tp/(2*tp+fp+fn)

    def compute_precision(self, arr1, arr2):     #NaN if prediction is normal
        overlap = np.multiply(arr1, arr2)
        
        tp = np.count_nonzero(overlap)
        fp = np.count_nonzero(arr2-overlap)
        
        if tp+fp == 0:
            return np.nan
        else:
            return tp/(tp+fp)

    def compute_recall(self, arr1, arr2):       #NaN if gt is normal
        overlap = np.multiply(arr1, arr2)
        
        tp = np.count_nonzero(overlap)
        fn = np.count_nonzero(arr1-overlap)
        
        if tp+fn == 0:
            return np.nan
        else:
            return tp/(tp+fn)

    def compute_FNR(self, arr1, arr2):
        overlap = np.multiply(arr1, arr2)
        
        tp = np.count_nonzero(overlap)
        fn = np.count_nonzero(arr1-overlap)

        if tp+fn == 0:
            return np.nan
        else:
            return fn/(tp+fn)

    def compute_specificity(self, gt, pred):
        overlap = np.multiply(gt, pred)
        union = gt + pred - overlap
        non_union = 1- union
        non_pred  = 1- pred
        
        tn = np.count_nonzero(np.multiply(non_union, non_pred))
        fp = np.count_nonzero(pred-overlap)
        
        if tn+fp == 0:
            return np.nan
        else:
            return tn/(tn+fp)