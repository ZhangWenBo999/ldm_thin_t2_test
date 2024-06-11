import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

import sys
sys.path.append("/kaggle/working/ldm_thin_t2_test")

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

# def make_batch(image, mask, device):
#     image = np.array(Image.open(image).convert("RGB"))
#
#     path = "/home/zwb/zwb/code/0531/ldm_thin/process_images/test/"
#     make_batch_image = image
#     # tensor_image = tensor_image.permute(0, 2, 3, 1)
#     # tensor_image = tensor_image.squeeze(0)
#     # make_batch_image = image.cpu().numpy()
#     Image.fromarray(np.uint8(make_batch_image)).save(path + f'make_batch_image.png')
#
#     image = image.astype(np.float32)/255.0
#     make_batch_image_0_1 = image
#     Image.fromarray(np.uint8(make_batch_image_0_1)).save(path + f'make_batch_image_0_1.png')
#
#     image = image[None].transpose(0,3,1,2)
#     image = torch.from_numpy(image)
#
#     mask = np.array(Image.open(mask).convert("L"))
#     make_batch_mask = mask
#     Image.fromarray(np.uint8(make_batch_mask)).save(path + f'make_batch_mask.png')
#
#     mask = mask.astype(np.float32)/255.0
#     make_batch_mask_0_1 = mask
#     Image.fromarray(np.uint8(make_batch_mask_0_1)).save(path + f'make_batch_mask_0_1.png')
#
#     mask = mask[None,None]
#     mask[mask < 0.5] = 0
#     mask[mask >= 0.5] = 1
#     mask = torch.from_numpy(mask)
#
#     masked_image = (1-mask)*image
#
#     make_batch_masked_image_0_1 = masked_image.clone().detach()
#     make_batch_masked_image_0_1 = make_batch_masked_image_0_1.permute(0, 2, 3, 1)
#     make_batch_masked_image_0_1 = make_batch_masked_image_0_1.squeeze(0)
#     make_batch_masked_image_0_1 = make_batch_masked_image_0_1.cpu().numpy()
#     Image.fromarray(np.uint8(make_batch_masked_image_0_1)).save(path + f'make_batch_masked_image_0_1.png')
#
#
#     batch = {"image": image, "mask": mask, "masked_image": masked_image}
#     for k in batch:
#         batch[k] = batch[k].to(device=device)
#         batch[k] = batch[k]*2.0-1.0
#     return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(glob.glob(os.path.join(opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load("/kaggle/working/ldm_thin_t2_test/configs/test_256.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("/kaggle/input/ldm-thin-t2-last9/last.ckpt")["state_dict"],
                          strict=False)

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = "cpu"
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                # path = "/home/zwb/zwb/code/0531/ldm_thin/process_images/test/"

                # batch_image_11_1 = batch['image'].clone().detach()
                # batch_image_11_1 = batch_image_11_1.permute(0, 2, 3, 1)
                # batch_image_11_1 = batch_image_11_1.squeeze(0)
                # batch_image_11_1 = batch_image_11_1.cpu().numpy()
                # Image.fromarray(np.uint8(batch_image_11_1)).save(path + f'batch_image_11_1.png')
                #
                # batch_mask_11_1 = batch['mask'].clone().detach()
                # batch_mask_11_1 = batch_mask_11_1.permute(0, 2, 3, 1)
                # batch_mask_11_1 = batch_mask_11_1.squeeze(0)
                # batch_mask_11_1 = batch_mask_11_1.squeeze(2)
                # batch_mask_11_1 = batch_mask_11_1.cpu().numpy()
                # Image.fromarray(np.uint8(batch_mask_11_1)).save(path + f'batch_mask_11_1.png')
                #
                # batch_masked_image_11_1 = batch['masked_image'].clone().detach()
                # batch_masked_image_11_1 = batch_masked_image_11_1.permute(0, 2, 3, 1)
                # batch_masked_image_11_1 = batch_masked_image_11_1.squeeze(0)
                # batch_masked_image_11_1 = batch_masked_image_11_1.cpu().numpy()
                # Image.fromarray(np.uint8(batch_masked_image_11_1)).save(path + f'batch_masked_image_11_1.png')


                # encode masked image and concat downsampled mask
                c = model.cond_stage_model.encode(batch["masked_image"])

                # c_11_1 = c.clone().detach()
                # c_11_1 = c_11_1.permute(0, 2, 3, 1)
                # c_11_1 = c_11_1.squeeze(0)
                # c_11_1 = c_11_1.cpu().numpy()
                # Image.fromarray(np.uint8(c_11_1)).save(path + f'c_11_1.png')

                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])

                # cc_11_1 = cc.clone().detach()
                # cc_11_1 = cc_11_1.permute(0, 2, 3, 1)
                # cc_11_1 = cc_11_1.squeeze(0)
                # cc_11_1 = cc_11_1.squeeze(2)
                # cc_11_1 = cc_11_1.cpu().numpy()
                # Image.fromarray(np.uint8(cc_11_1)).save(path + f'cc_11_1.png')

                c = torch.cat((c, cc), dim=1)

                # c_cc_11_1 = c.clone().detach()
                # c_cc_11_1 = c_cc_11_1.permute(0, 2, 3, 1)
                # c_cc_11_1 = c_cc_11_1.squeeze(0)
                # c_cc_11_1 = c_cc_11_1.cpu().numpy()
                # Image.fromarray(np.uint8(c_cc_11_1)).save(path + f'c_cc_11_1.png')

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)

                # ddim_sample = samples_ddim.clone().detach()
                # ddim_sample = ddim_sample.permute(0, 2, 3, 1)
                # ddim_sample = ddim_sample.squeeze(0)
                # ddim_sample = ddim_sample.cpu().numpy()
                # Image.fromarray(np.uint8(ddim_sample)).save(path + f'ddim_sample.png')

                x_samples_ddim = model.decode_first_stage(samples_ddim)

                # decode_sample = x_samples_ddim.clone().detach()
                # decode_sample = decode_sample.permute(0, 2, 3, 1)
                # decode_sample = decode_sample.squeeze(0)
                # decode_sample = decode_sample.cpu().numpy()
                # Image.fromarray(np.uint8(decode_sample)).save(path + f'decode_sample.png')

                image = torch.clamp((batch["image"]+1.0)/2.0,
                                    min=0.0, max=1.0)

                # decode_image = image.clone().detach()
                # decode_image = decode_image.permute(0, 2, 3, 1)
                # decode_image = decode_image.squeeze(0)
                # decode_image = decode_image.cpu().numpy()
                # Image.fromarray(np.uint8(decode_image)).save(path + f'decode_image.png')

                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)

                # decode_mask = mask.clone().detach()
                # decode_mask = decode_mask.permute(0, 2, 3, 1)
                # decode_mask = decode_mask.squeeze(0)
                # decode_mask = decode_mask.squeeze(2)
                # decode_mask = decode_mask.cpu().numpy()
                # Image.fromarray(np.uint8(decode_mask)).save(path + f'decode_mask.png')

                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                # decode_predicted_image = predicted_image.clone().detach()
                # decode_predicted_image = decode_predicted_image.permute(0, 2, 3, 1)
                # decode_predicted_image = decode_predicted_image.squeeze(0)
                # decode_predicted_image = decode_predicted_image.cpu().numpy()
                # Image.fromarray(np.uint8(decode_predicted_image)).save(path + f'decode_predicted_image.png')

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)

                # Image.fromarray(inpainted.astype(np.uint8)).save(path + f'inpainted.png')
