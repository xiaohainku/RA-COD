import faiss
import torch
import torchvision.transforms as T
from PIL import Image
from dinov2.hub.backbones import dinov2_vitl14
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from segment_anything_hq import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from utils import get_multi_bboxes, EvaluationMetricsV2, save_evaluation_hqsam
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", type=int, default=518)
parser.add_argument("--test_dataset_path", type=str, default=None, help="path to test dataset")
parser.add_argument("--dino_ckpt_path", type=str, default=None, help="path to dino checkpoint")
parser.add_argument("--sam_ckpt_path", type=str, default=None, help="path to sam checkpoint")
parser.add_argument("--prototype_path", type=str, default=None, help="path to prototype")
parser.add_argument("--mask_save_path", type=str, default=None, help="path to save masks")
parser.add_argument("--device", type=str, default='cuda')

args = parser.parse_args()

embed_dim = 1024
patch_size = int(args.img_size / 14)
sam_infer_iter = 2

metric_save_path = os.path.join(args.mask_save_path, 'metric.yaml')

fore_proto = np.load(f'{args.prototype_path}/fore.npy')
back_proto = np.load(f'{args.prototype_path}/back.npy')
envir_proto = np.load(f'{args.prototype_path}/envir.npy')

fore_prototype_num = fore_proto.shape[0]

res = faiss.StandardGpuResources()  # use a single GPU
index_flat = faiss.IndexFlatIP(fore_proto.shape[1])
index = faiss.index_cpu_to_gpu(res, 0, index_flat)

faiss.normalize_L2(fore_proto)
index.add(fore_proto)
faiss.normalize_L2(back_proto)
index.add(back_proto)
faiss.normalize_L2(envir_proto)
index.add(envir_proto)

transform = T.Compose([
    T.Resize((args.img_size, args.img_size)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

dinov2 = dinov2_vitl14(ckpt_path=args.dino_ckpt_path).to(args.device)

sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path).to(args.device)
predictor = SamPredictor(sam)

datasets = ['chameleon', 'camo', 'cod10k', 'nc4k']

for dataset in datasets:
    metric_key = f'{dataset.upper()}'
    print(metric_key)

    if os.path.exists(metric_save_path):
        with open(metric_save_path, 'r') as f:
            dt = yaml.load(f, Loader=yaml.FullLoader)
    else:
        dt = dict()

    if metric_key in dt:
        continue

    source_dir = f'{args.test_dataset_path}/{dataset.upper()}/Imgs'
    gt_source_dir = f'{args.test_dataset_path}/{dataset.upper()}/GT'
    mask_save_dir = f'{args.mask_save_path}/{dataset.upper()}'

    os.makedirs(mask_save_dir, exist_ok=True)

    imgs = [i for i in os.listdir(source_dir) if i.endswith(".jpg")]
    metric = EvaluationMetricsV2()
    metric_dt = dict()
    metric_dict = dict()

    for img_name in tqdm(imgs):
        img_path = os.path.join(source_dir, img_name)
        gt_path = os.path.join(gt_source_dir, img_name.split('.')[0] + '.png')
        mask_save_path = os.path.join(mask_save_dir, img_name.split('.')[0] + '.png')

        img = Image.open(img_path).convert('RGB')

        width, height = img.size
        img_tensor = transform(img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            feats = dinov2.get_intermediate_layers(img_tensor, reshape=True)[0]
        query = feats.permute(2, 3, 1, 0).reshape(patch_size * patch_size, embed_dim, -1).contiguous().squeeze().cpu().numpy()
        faiss.normalize_L2(query)

        _, index_knn = index.search(query, 1)

        coarse_mask = np.sum(index_knn < fore_prototype_num, axis=1)
        coarse_mask = coarse_mask.astype(np.float32).reshape(patch_size, patch_size)
        coarse_mask = (coarse_mask > 0.5).astype(np.float32)

        coarse_mask = F.interpolate(torch.tensor(coarse_mask).unsqueeze(0).unsqueeze(0), size=(height, width),
                                 mode='bilinear', align_corners=True).squeeze().numpy()

        predictor.set_image(np.array(img.convert('RGB')))

        input_boxes = get_multi_bboxes(coarse_mask)

        batch_box = False if input_boxes is None else len(input_boxes) > 1
        if batch_box:
            input_boxes = torch.tensor(input_boxes, device=predictor.device)
            transformed_box = predictor.transform.apply_boxes_torch(input_boxes, (height, width))
            input_points = None
            labels = None,
        logits = None
        for i in range(sam_infer_iter):
            if not batch_box:
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.array(input_boxes[0]),
                    multimask_output=False,
                    hq_token_only=True,
                    mask_input=logits,
                )
                if i == sam_infer_iter - 1:
                    masks = torch.tensor(masks, dtype=torch.float32, device=args.device).unsqueeze(0)
            else:
                masks, scores, logits = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_box,
                    multimask_output=False,
                    hq_token_only=False,
                    mask_input=logits,
                )
                if i == sam_infer_iter - 1:
                    masks = masks.to(dtype=torch.float32)

        scale_mask = F.interpolate(masks, size=(patch_size, patch_size), mode='bilinear', align_corners=True)
        mask_query = (torch.sum(feats * scale_mask, dim=(2, 3)) / (
                torch.sum(scale_mask > 0, dim=(2, 3)) + 1e-8)).cpu().numpy().astype('float32')
        faiss.normalize_L2(mask_query)
        _, mask_index = index.search(mask_query, 1)

        a = np.sum(mask_index < fore_prototype_num, axis=1)
        a = (a > 0.5).astype(np.uint8)

        final_mask = (np.sum(np.expand_dims(a, axis=(1, 2, 3)) * masks.cpu().numpy(), axis=(0, 1)) > 0).astype(np.uint8)

        save_evaluation_hqsam(final_mask, mask_save_path, gt_path, metric, save_mask=True)

    metric_dic = metric.get_results()

    metric_dict['sm'] = sm = float(metric_dic['sm'])
    metric_dict['emMean'] = emMean = float(metric_dic['emMean'])
    metric_dict['emAdp'] = emAdp = float(metric_dic['emAdp'])
    metric_dict['emMax'] = emMax = float(metric_dic['emMax'])
    metric_dict['fmMean'] = fmMean = float(metric_dic['fmMean'])
    metric_dict['fmAdp'] = fmAdp = float(metric_dic['fmAdp'])
    metric_dict['fmMax'] = fmMax = float(metric_dic['fmMax'])
    metric_dict['wfm'] = wfm = float(metric_dic['wfm'])
    metric_dict['mae'] = mae = float(metric_dic['mae'])

    metric_dt[metric_key] = metric_dict
    with open(metric_save_path, 'a') as f:
        yaml.dump(metric_dt, f)

    print('sm:', sm)
    print('emMean:', emMean)
    print('emAdp:', emAdp)
    print('emMax:', emMax)
    print('fmMean:', fmMean)
    print('fmAdp:', fmAdp)
    print('fmMax:', fmMax)
    print('wfm:', wfm)
    print('mae:', mae)