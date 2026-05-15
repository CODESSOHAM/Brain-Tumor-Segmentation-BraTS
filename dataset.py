import os
import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, 
    Orientationd, ScaleIntensityRanged, CropForegroundd, 
    RandCropByPosNegLabeld, ToTensord
)
from monai.data import Dataset, DataLoader

def get_loader(data_dir, batch_size=2):
    images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label",
            spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4,
            image_key="image", image_threshold=0,
        ),
        ToTensord(keys=["image", "label"]),
    ])

    ds = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)