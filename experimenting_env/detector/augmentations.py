import albumentations as A
from albumentations.pytorch import ToTensorV2

switch = {
    "none": [
        ToTensorV2(),
    ],
    "bbs_crop": [
        A.RandomSizedBBoxSafeCrop(480, 640),

        ToTensorV2(),
    ],
    "bbs_crop_strong": [
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(),
        A.RandomSizedBBoxSafeCrop(480, 640),
        ToTensorV2(),
    ],
    "bbs_crop_strong2": [
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(),
        A.RandomCrop(480, 640),
        ToTensorV2(),
    ],
    "strong_image": [
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(),
        ToTensorV2(),
    ],
}


def get_transform(transform_type):
    if transform_type in switch:
        return switch[transform_type]
    return switch["none"]
