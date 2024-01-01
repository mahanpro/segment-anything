import torch
import warnings
from segment_anything.utils.onnx import SamOnnxModel
from segment_anything import sam_model_registry, SamPredictor
import pandas as pd


def Initialize():
    img_path              = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_raw/Dataset086_TCIA/imagesTr"
    gt_path               = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/nnUNetv2_files/nnUNet_raw/Dataset086_TCIA/labelsTr"
    studies_names_path    = r"/storage/nfs_createproject_tciadataset/FDG-PET-CT-Lesions/Processing/Mahan_codes_docus/nnUnetFrame/list_of_renamed_studies_whole_dataset.csv"
    sam_checkpoint  = "/home/pouromm/segment-anything/model_checkpoint/sam_vit_h_4b8939.pth"
    onnx_model_path = "/home/pouromm/segment-anything/exported_onnx_model/onnx_model.onnx"

    #################### Loading the Dataset ####################
    Studies_name_dict = pd.read_csv(studies_names_path, header=None)
    Studies_name_dict = dict(Studies_name_dict)
    Studies_name_dict = dict(zip(Studies_name_dict[0],Studies_name_dict[1]))
    #############################################################
    device         = "cuda"
    model_type     = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
    }

    embed_dim       = sam.prompt_encoder.embed_dim
    embed_size      = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs    = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size,          dtype=torch.float),
        "point_coords"    : torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels"    : torch.randint(low=0, high=4, size=(1, 5),       dtype=torch.float),
        "mask_input"      : torch.randn(1, 1, *mask_input_size,             dtype=torch.float),
        "has_mask_input"  : torch.tensor([1],                               dtype=torch.float),
        "orig_im_size"    : torch.tensor([1500, 2250],                      dtype=torch.float),
    }

    output_names = ["masks", "iou_predictions", "low_res_masks"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    sam.to(device=device)
    predictor = SamPredictor(sam)
    return img_path, gt_path, predictor, Studies_name_dict, onnx_model_path