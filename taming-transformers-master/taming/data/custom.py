import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import torch
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

from PIL import Image
from mask_generation.utils import MaskGeneration, MergeMask, RandomAttribute

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    # image = image[None].transpose(0,3,1,2)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)

    # mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    # mask = mask[None,None]
    mask = mask[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]

        array = np.array(Image.open(example['file_path_']))

        mode = RandomAttribute(self.mask_mode, 256)
        mask_generation = MaskGeneration()
        mask = mask_generation(array, mode, verbose=True)

        batch = make_batch(example['file_path_'], mask, device='cuda' if torch.cuda.is_available() else 'cpu')
        example['mask'] = batch['mask'].permute(1, 2, 0).cpu().numpy()
        example['masked_image'] = batch['masked_image'].permute(1, 2, 0).cpu().numpy()

        # path = "/home/zwb/zwb/code/0531/ldm_thin/process_images/train1/"
        # batch_image_11_1 = example['image']
        # Image.fromarray(np.uint8(batch_image_11_1)).save(path + f'batch_image_11_1.png')
        #
        # batch_mask_11_1 = example['mask']
        # batch_mask_11_1 = torch.from_numpy(batch_mask_11_1)
        # batch_mask_11_1 = batch_mask_11_1.clone().detach()
        # # batch_mask_11_1 = batch_mask_11_1.permute(0, 2, 3, 1)
        # # batch_mask_11_1 = batch_mask_11_1.squeeze(0)
        # batch_mask_11_1 = batch_mask_11_1.squeeze(2)
        # batch_mask_11_1 = batch_mask_11_1.cpu().numpy()
        # Image.fromarray(np.uint8(batch_mask_11_1)).save(path + f'batch_mask_11_1.png')
        #
        # batch_masked_image_11_1 = example['masked_image']
        # Image.fromarray(np.uint8(batch_masked_image_11_1)).save(path + f'batch_masked_image_11_1.png')

        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, mask_mode):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.mask_mode = mask_mode


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, mask_mode):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.mask_mode = mask_mode


