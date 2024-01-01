import torch
import numpy as np
import cv2
import os
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime
import matplotlib.pyplot as plt
from onnxruntime.quantization import QuantType
import Utils
from Initialize import Initialize

img_path, gt_path, predictor, Studies_name_dict, onnx_model_path= Initialize()
ort_session = onnxruntime.InferenceSession(onnx_model_path)

def perform_predction(filename, combination, directory, Dilation):
    filename_object = Utils.study(filename)
    imagePET = nib.load(os.path.join(img_path, filename + "_0001.nii.gz")).get_fdata()
    image_strechedPET = Utils.contrast_stretching(imagePET)
    if not combination == "3PET":
        imageCT  = nib.load(os.path.join(img_path, filename + "_0000.nii.gz")).get_fdata()
        imageCT_preprocessed = Utils.preprocess_CT(imageCT)
    nnUnet_pred = Utils.proper_prediction(filename + ".nii.gz")
    gt          = nib.load(os.path.join(gt_path, filename + ".nii.gz")).get_fdata()

    if Dilation:
        if nnUnet_pred.any():
            nnUnet_pred = Utils.random_dilation(nnUnet_pred)

    final_output = np.zeros((nnUnet_pred.shape), dtype=np.float32)
    for slc in range(nnUnet_pred.shape[2]):
        if nnUnet_pred[:, :, slc].any():
            if combination == "3PET":
                predictor.set_image(np.repeat(image_strechedPET[:, :, slc][:, :, None], 3, axis=2).astype('uint8'))
            elif combination == "2PET1CT":
                im = np.repeat(image_strechedPET[:, :, slc][:, :, None], 3, axis=2)
                im[:, :, 1] = imageCT_preprocessed[:, :, slc]
                predictor.set_image(im.astype('uint8'))
            elif combination == "2CT1PET":
                im = np.repeat(imageCT_preprocessed[:, :, slc][:, :, None], 3, axis=2)
                im[:, :, 1] = image_strechedPET[:, :, slc]
                predictor.set_image(im.astype('uint8'))
            else:
                raise ValueError("modality should be either 3PET, 2PET1CT or 2CT1PET")
            onnx_box_coords, onnx_box_labels = Utils.export_contours_to_onnx_model(nnUnet_pred[:, :, slc])
            onnx_box_coords     = predictor.transform.apply_coords(onnx_box_coords, nnUnet_pred.shape[:2]).astype(np.float32)
            onnx_mask_input     = np.zeros((1, 1, 256, 256), dtype=np.float32)
            onnx_has_mask_input = np.zeros(1, dtype=np.float32)
            image_embedding     = predictor.get_image_embedding().cpu().numpy()

            ort_inputs = {
                "image_embeddings": image_embedding,
                "point_coords"    : onnx_box_coords,
                "point_labels"    : onnx_box_labels,
                "mask_input"      : onnx_mask_input,
                "has_mask_input"  : onnx_has_mask_input,
                "orig_im_size"    : np.array(nnUnet_pred.shape[:2], dtype=np.float32)
            }

            mask, _, low_res_logits = ort_session.run(None, ort_inputs)
            mask = mask > predictor.model.mask_threshold
            final_output[:, :, slc] = mask[0, 0, :, :]

    filename_object.dice      = filename_object.compute_dice(gt, final_output)
    filename_object.recall    = filename_object.compute_recall(gt, final_output)
    filename_object.precision = filename_object.compute_precision(gt, final_output)
    filename_object.FNR       = filename_object.compute_FNR(gt, final_output)
    filename_object.specifity = filename_object.compute_specificity(gt, final_output)


    new_image = nib.Nifti1Image(final_output, affine=np.eye(4))
    new_image.header.get_xyzt_units()
    new_image.to_filename(os.path.join(directory, os.path.join(combination, filename + ".nii.gz")))

    return filename_object.dice, filename_object.recall, filename_object.precision, filename_object.FNR, filename_object.specifity

combinations = ["3PET", "2PET1CT", "2CT1PET"]
output_folder = r"/home/pouromm/SAM_with_nnUNet_prompts_results"

for combination in combinations:

    #### Without dilation
    Dilation = False
    Dice_list      = []
    Recall_list    = []
    Precision_list = []
    FNR_list       = []
    Specifity_list = []
    patient_ID     = []

    for filename, _ in Studies_name_dict.items():
        dice, recall, precision, FNR, specificity = perform_predction(filename, combination, output_folder, Dilation)

        Dice_list.append(dice)
        Recall_list.append(recall)
        Precision_list.append(precision)
        FNR_list.append(FNR)
        Specifity_list.append(specificity)
        patient_ID.append(filename)

    d = {'Patient ID': patient_ID, 'Dice': Dice_list, 'Recall': Recall_list, 'Precision': Precision_list, 'FNR': FNR_list, 'Specifity': Specifity_list}
    dataFrame = pd.DataFrame(data=d)
    dataFrame.to_csv(os.path.join(output_folder, "SAM_with_nnUNet_prompts_results_without_dilation_" + str(combination) + ".csv"), sep=',', index=False, encoding='utf-8')

    #### With dilation
    Dilation = True
    Dice_list      = []
    Recall_list    = []
    Precision_list = []
    FNR_list       = []
    Specifity_list = []
    patient_ID     = []

    for filename, _ in Studies_name_dict.items():
        dice, recall, precision, FNR, specificity = perform_predction(filename, combination, output_folder, Dilation)

        Dice_list.append(dice)
        Recall_list.append(recall)
        Precision_list.append(precision)
        FNR_list.append(FNR)
        Specifity_list.append(specificity)
        patient_ID.append(filename)

    d = {'Patient ID': patient_ID, 'Dice': Dice_list, 'Recall': Recall_list, 'Precision': Precision_list, 'FNR': FNR_list, 'Specifity': Specifity_list}
    dataFrame = pd.DataFrame(data=d)
    dataFrame.to_csv(os.path.join(output_folder, "SAM_with_nnUNet_prompts_results_with_dilation_" + str(combination) + ".csv"), sep=',', index=False, encoding='utf-8')

        