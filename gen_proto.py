import torch
import torchvision.transforms as T
from PIL import Image
from dinov2.hub.backbones import dinov2_vitl14
import os
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=518)
parser.add_argument("--gen_data_path", type=str, default=None, help="path to generated data")
parser.add_argument("--dino_ckpt_path", type=str, default=None, help="path to dino checkpoint")
parser.add_argument("--prototype_save_path", type=str, default=None, help="path to save prototype")
parser.add_argument("--fore_proto_num_per_cate", type=int, default=64)
parser.add_argument("--envir_proto_num_per_cate", type=int, default=32)
parser.add_argument("--device", type=str, default='cuda')

args = parser.parse_args()

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

embed_dim = 1024
patch_size = int(args.img_size / 14)

dinov2 = dinov2_vitl14(ckpt_path=args.dino_ckpt_path).to(args.device)

transform = T.Compose([
    T.Resize(args.img_size, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

mask_transform = T.Compose([
    T.Resize(patch_size, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor()])

## foreground prototype
print("Constructing foreground prototypes...")

cate_ls = os.listdir(f"{args.gen_data_path}/foreground/img")
fore_prototype = np.zeros((args.fore_proto_num_per_cate * len(cate_ls), embed_dim), dtype='float32')
back_prototype = np.zeros((args.fore_proto_num_per_cate * len(cate_ls), embed_dim), dtype='float32')

for i, cate in tqdm(enumerate(cate_ls), total=len(cate_ls)):

    cate_img_path = os.path.join(f"{args.gen_data_path}/foreground/img", cate)
    cate_mask_path = os.path.join(f"{args.gen_data_path}/foreground/mask", cate)

    imgs = [i for i in os.listdir(cate_img_path) if i.endswith(".jpg")]
    imgs = sorted(imgs)

    for j, img_name in enumerate(imgs[:args.fore_proto_num_per_cate]):
        img_path = os.path.join(cate_img_path, img_name)
        mask_path = os.path.join(cate_mask_path, img_name.split('.')[0] + '.png')

        img = Image.open(img_path)
        mask = binary_loader(mask_path)

        img_tensor = transform(img).unsqueeze(0).to(args.device)
        mask_tensor = mask_transform(mask).unsqueeze(0).to(args.device)

        with torch.no_grad():
            feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]

        fore_feat = feats * mask_tensor
        back_feat = feats * (1 - mask_tensor)
        fore = (torch.sum(fore_feat, dim=(2, 3)) / torch.sum(fore_feat > 0)).cpu().numpy()
        back = (torch.mean(back_feat, dim=(2, 3)) / torch.sum(back_feat > 0)).cpu().numpy()

        fore_prototype[i * args.fore_proto_num_per_cate + j, :] = fore
        back_prototype[i * args.fore_proto_num_per_cate + j, :] = back

os.makedirs(args.prototype_save_path, exist_ok=True)
np.save(f'{args.prototype_save_path}/fore.npy', fore_prototype)
np.save(f'{args.prototype_save_path}/back.npy', back_prototype)


## environment prototype
print("Constructing environment prototypes...")

envir_cate_ls = os.listdir(f"{args.gen_data_path}/background/img")

envir_prototype = np.zeros((args.envir_proto_num_per_cate * len(envir_cate_ls), embed_dim), dtype='float32')

for i, cate in tqdm(enumerate(envir_cate_ls), total=len(envir_cate_ls)):

    cate_img_path = os.path.join(f"{args.gen_data_path}/background/img", cate)
    imgs = [i for i in os.listdir(cate_img_path) if i.endswith(".jpg")]
    imgs = sorted(imgs)

    for j, img_name in enumerate(imgs[:args.envir_proto_num_per_cate]):
        img_path = os.path.join(cate_img_path, img_name)

        img = Image.open(img_path)

        img_tensor = transform(img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]

        envir = torch.mean(feats, dim=(2, 3)).cpu().numpy()

        envir_prototype[i * args.envir_proto_num_per_cate + j, :] = envir

np.save(f'{args.prototype_save_path}/envir.npy', envir_prototype)