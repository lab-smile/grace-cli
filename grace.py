#!/usr/bin/env python3
import os
import sys
import json
import torch
import logging
import nibabel as nib
from time import sleep
from scipy.io import savemat
from monai.data import MetaTensor, DataLoader, Dataset, load_decathlon_datalist
from monai.networks.nets import UNETR
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Spacingd, Orientationd, ScaleIntensityRanged, EnsureTyped, LoadImaged, EnsureChannelFirstd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

logger=logging.getLogger(__name__)


# HOW TO USE:
# logger.info("Created output directories.")

def send_progress(message, progress):
    """
        Helper function to print SSE progress updates
        @param message: Message about current stage of model prediction (str)
        @param progress: Progress percentage (int)
    """
    # data = json.dumps({"message": message, "progress": progress})
    # return f"data: {data}\n\n"
    logger.info(f"{message}... {progress}%")

def generate_datalist(folder_path):
    nii_files = [
        {"image": os.path.abspath(os.path.join(folder_path, f))}
        for f in os.listdir(folder_path)
        if f.endswith(".nii.gz") or f.endswith(".nii")
    ]

    datalist = {"testing": nii_files}

    with open("datalist.json", "w") as f:
        json.dump(datalist, f, indent=4)

def load_model(model_path, spatial_size, num_classes, device, dataparallel=False, num_gpu=1):
    """
        Load and configure the model for inference.
        @param model_path: Path to the model weights file (str)
        @param spatial_size: Size of the input images (tuple)
        @param num_classes: Number of output classes (int)
        @param device: Device to run the model on (str or torch.device)
        @param dataparallel: Whether to use DataParallel (bool)
        @param num_gpu: Number of GPUs to use if dataparallel is True (int)
        @return: Configured model for inference (torch.nn.Module)
    """
    send_progress("Configuring model...", 10)

    model = UNETR(
        in_channels=1,
        out_channels=num_classes,
        img_size=spatial_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        proj_type="perceptron",
    )

    # if dataparallel:
    #     yield send_progress("Initializing DataParallel with multiple GPUs", 15)
    #     model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))

    model = model.to(device)
    send_progress(f"Loading model weights from {model_path}...", 20)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    send_progress("Model loaded successfully.", 25)
    return model


def preprocess_datalists(a_min, a_max, target_shape=(64, 64, 64)):
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="trilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"], track_meta=True)
    ])

def preprocess_input(input_path, device, a_min_value, a_max_value):
    """
        Load and preprocess the input NIfTI image.
        @param input_path: Path to the input NIfTI image file (str)
        @param device: Device to run the preprocessing on (str or torch.device)
        @param a_min_value: Minimum intensity value for scaling (int or float)
        @param a_max_value: Maximum intensity value for scaling (int or float)
    """
    send_progress(f"Loading input image from {input_path}...", 30)
    input_img = nib.load(input_path)
    image_data = input_img.get_fdata()
    send_progress(f"Input image loaded. Shape: {image_data.shape}", 35)

    # Convert to MetaTensor for MONAI compatibility
    meta_tensor = MetaTensor(image_data, affine=input_img.affine)

    send_progress("Applying preprocessing transforms...", 40)
    
    # Apply MONAI test transforms
    test_transforms = Compose(
        [
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("trilinear"),
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=a_min_value, a_max=a_max_value, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    data = {"image": meta_tensor}
    transformed_data = test_transforms(data)

    # Convert to PyTorch tensor
    image_tensor = transformed_data["image"].clone().detach().unsqueeze(0).unsqueeze(0).to(device)
    send_progress(f"Preprocessing complete. Model input shape: {image_tensor.shape}", 45)
    return image_tensor, input_img

def save_predictions(predictions, input_img, output_dir, base_filename):
    """
        Save predictions as NIfTI and MAT files.
        @param predictions: Model output predictions (torch.Tensor)
        @param input_img: Original input image used for predictions (nibabel Nifti1Image)
        @param output_dir: Directory to save the output files (str)
        @param base_filename: Base filename for the saved output files (str)
    """
    send_progress("Post-processing predictions...", 80)
    processed_preds = torch.argmax(predictions, dim=1).detach().cpu().numpy().squeeze()
    
    # Save as .nii.gz
    send_progress("Saving NIfTI file...", 85)
    pred_img = nib.Nifti1Image(processed_preds, affine=input_img.affine, header=input_img.header)
    nii_save_path = os.path.join(output_dir, f"{base_filename}_pred_GRACE.nii.gz")
    nib.save(pred_img, nii_save_path)
    
    # Save as .mat
    send_progress("Saving MAT file...", 90)
    mat_save_path = os.path.join(output_dir, f"{base_filename}_pred_GRACE.mat")
    savemat(mat_save_path, {"testimage": processed_preds})
    send_progress("Files saved successfully.", 95)

def save_multiple_predictions(predictions, batch_meta, output_dir):
    """
        Save predictions as NIfTI and MAT files.
        @param predictions: Model output predictions (torch.Tensor)
        @param input_img: Original input image used for predictions (nibabel Nifti1Image)
        @param output_dir: Directory to save the output files (str)
        @param base_filename: Base filename for the saved output files (str)
    """
    send_progress("Post-processing predictions...", 80)
    for i in range(predictions.shape[0]):
        pred_np = torch.argmax(predictions[i], dim=0).cpu().numpy().squeeze()
        
        filename = os.path.basename(batch_meta["filename_or_obj"][i]).replace(".nii.gz", "")
        affine = batch_meta["affine"][i].numpy()
        header = nib.load(batch_meta["filename_or_obj"][i]).header

        # Save as .nii.gz
        send_progress(f"Processing output for {i}th input file...", 85)
        nib.save(nib.Nifti1Image(pred_np, affine, header), os.path.join(output_dir, f"{filename}_pred_GRACE.nii.gz"))

        savemat(os.path.join(output_dir, f"{filename}_pred_GRACE.mat"), {"testimage": pred_np})
    send_progress("Files saved successfully.", 95)

def grace_predict_single_file(input_path, output_dir="output", model_path="models/GRACE.pth",
                       spatial_size=(64, 64, 64), num_classes=12, dataparallel=False, num_gpu=1,
                       a_min_value=0, a_max_value=255):
    """
        Predict segmentation for a single NIfTI image with progress updates via SSE.
        @param input_path: Path to the input NIfTI image file (str)
        @param output_dir: Directory to save the output files (str)
        @param model_path: Path to the model weights file (str)
        @param spatial_size: Size of the input images (tuple)
        @param num_classes: Number of output classes (int)
        @param dataparallel: Whether to use DataParallel (bool)
        @param num_gpu: Number of GPUs to use if dataparallel is True (int)
        @param a_min_value: Minimum intensity value for scaling (int or float)
        @param a_max_value: Maximum intensity value for scaling (int or float)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(input_path).split(".nii")[0]

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("cpu")
        send_progress("Using MPS backend (CPU due to ConvTranspose3d support limitations)", 5)
    else:
        send_progress(f"Using device: {device}", 5)

    # Load model
    model = load_model(model_path, spatial_size, num_classes, device, dataparallel, num_gpu)

    # Preprocess input
    image_tensor, input_img = preprocess_input(input_path, device, a_min_value, a_max_value)

    # Perform inference
    send_progress("Starting sliding window inference...", 50)
    with torch.no_grad():
        predictions = sliding_window_inference(
            image_tensor, spatial_size, sw_batch_size=4, predictor=model, overlap=0.8
        )
    send_progress("Inference completed successfully.", 75)

    # Save predictions
    save_predictions(predictions, input_img, output_dir, base_filename)
    
    send_progress("Processing completed successfully!", 100)

def grace_predict_multiple_files(input_path, output_dir="output", model_path="models/GRACE.pth",
                       spatial_size=(64, 64, 64), num_classes=12, dataparallel=False, num_gpu=1,
                       a_min_value=0, a_max_value=255):
    """
        Predict segmentation for a single NIfTI image with progress updates via SSE.
        @param input_path: Path to the input NIfTI image file (str)
        @param output_dir: Directory to save the output files (str)
        @param model_path: Path to the model weights file (str)
        @param spatial_size: Size of the input images (tuple)
        @param num_classes: Number of output classes (int)
        @param dataparallel: Whether to use DataParallel (bool)
        @param num_gpu: Number of GPUs to use if dataparallel is True (int)
        @param a_min_value: Minimum intensity value for scaling (int or float)
        @param a_max_value: Maximum intensity value for scaling (int or float)
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 10
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("cpu")
        send_progress("Using MPS backend (CPU due to ConvTranspose3d support limitations)", 5)
    else:
        send_progress(f"Using device: {device}", 5)

    datalist = load_decathlon_datalist(input_path, True, "testing")
    transforms = preprocess_datalists(a_min_value, a_max_value, target_shape=spatial_size)
    dataset = Dataset(data=datalist, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    # Load model
    model = load_model(model_path, spatial_size, num_classes, device, dataparallel, num_gpu)

    # Perform inference
    send_progress("Starting sliding window inference...", 50)
    for batch in dataloader:
        print(batch.keys()) 
        print(batch.get("image_meta_dict", "NO META FOUND"))
        images = batch["image"].to(device)
        meta = batch["image"].meta

        with torch.no_grad():
            preds = sliding_window_inference(
                images, spatial_size, sw_batch_size=1, predictor=model, overlap=0.8
            )
        save_multiple_predictions(preds, meta, output_dir)
    
    send_progress("Processing completed successfully!", 100)

# Example usage
if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Path for input file or a folder expected!")
    elif(len(sys.argv) > 3):
        print("Too many arguments...!")

    input_path = sys.argv[1]
    output_dir = "outputs"
    model_path = "GRACE.pth"
    datalist_path = "datalist.json"

    if os.path.isdir(input_path):
        print("Generating datalist...")
        generate_datalist(input_path)

        grace_predict_multiple_files(
            input_path=datalist_path,
            output_dir=output_dir,
            model_path=model_path,
            spatial_size=(64, 64, 64),
            num_classes=12,
            dataparallel=False,
            num_gpu=1,
            a_min_value=0,
            a_max_value=255,    
        )
    
    else:
        grace_predict_single_file(
            input_path=input_path,
            output_dir=output_dir,
            model_path=model_path,
            spatial_size=(64, 64, 64),
            num_classes=12,
            dataparallel=False,
            num_gpu=1,
            a_min_value=0,
            a_max_value=255,
        )

    print("Output files generated...!")