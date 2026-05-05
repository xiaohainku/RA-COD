python gen_proto.py \
--gen_data_path /data/dataset/RA-COD/main_paper/generated_data \
--dino_ckpt_path /data/download/ckpt/dino/dinov2_vitl14_pretrain.pth \
--prototype_save_path /data/dataset/RA-COD/main_paper/generated_data/prototype

python retrieval.py \
--test_dataset_path /data/dataset/UpGen/dataset/TestDataset \
--dino_ckpt_path /data/download/ckpt/dino/dinov2_vitl14_pretrain.pth \
--sam_ckpt_path /data/download/ckpt/sam/sam_hq_vit_h.pth \
--prototype_path /data/dataset/RA-COD/main_paper/generated_data/prototype \
--mask_save_path /data/dataset/RA-COD/main_paper/predicted_maps