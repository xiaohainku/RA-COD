# RA-COD: Retrieval-Augmented Camouflaged Object Detection

> [**RA-COD: Retrieval-Augmented Camouflaged Object Detection**](https://github.com/xiaohainku/RA-COD)
>
> **TIP 2026**
> 
> [Ji Du](https://xiaohainku.github.io/), [Jiesheng Wu](https://scholar.google.com/citations?hl=en&user=fDgrxRcAAAAJ), Desheng Kong, Fangwei Hao, [Jing Xu](https://ai.nankai.edu.cn/info/1033/3528.htm), [Ping Li](https://scholar.google.com/citations?user=mQ9YyHsAAAAJ&hl=en)
>
> **NKU & PolyU**

**Abstract**

Camouflaged Object Detection (COD) is pivotal for segmenting objects that seamlessly blend into their surroundings. While prior endeavors demonstrate impressive performance through training on predefined labels, they heavily rely on labor-intensive data annotation and struggle to adapt to open-world scenarios. In this light, we propose RA-COD, a training-free paradigm that enables COD by retrieving the most similar samples from the prototype repository. The efficacy of RA-COD hinges on (1) capturing the nuanced resemblance between objects and their environments and (2) excelling in dense prediction tasks. To achieve (1), the crux lies in ensuring diversity and discriminability within the prototype repository. In this context, we propose GenPro, an automated pipeline for crafting Generative Prototypes. GenPro integrates a range of foundation models, including the Diffusion Model, Vision-Language Model, Segment Anything Model (SAM), and DINOv2, in a complementary manner that synergistically generates diverse and distinguishable prototype samples. To achieve (2), we propose C2F to retrieve camouflaged objects in a Coarse-to-Fine regime. We commence with pixel-level retrieval in the feature space, which generates a coarse mask that effectively captures class discrimination and object localization. Further refinement is achieved by extracting bounding boxes from this coarse mask to prompt SAM in generating mask proposals for region-level retrieval. Evaluations on four benchmarks showcase that RA-COD achieves state-of-the-art performance compared to existing training-free methods.

<p float="left">
  <img src="figs/overview.png?raw=true" width="100%" />
</p>


## **Preparation**

1. **Environment**

   ```shell
   pip install -r requirement.txt
   ```

2. **Foundation model checkpoints**

   - DINOv2: [DINOv2-ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)

   - SAM: [HQSAM-ViT-H](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing)

3. **Data**

   - test dataset (CHAMELEON / CAMO / COD10K / NC4K)

   - generated data (image / attention map / mask / prototype)

   - predicted maps

   are availabel at: [Google](https://drive.google.com/drive/folders/1ASiUaVflRLIMlBH1k1WKafCUvhMRDd5d?usp=sharing) | [Baidu](https://pan.baidu.com/s/1t5xD96z8n-K0uJNrCeoU6w?pwd=qnqa)

## Evaluation

1. **Quickly reproduce the original results**

   - download our precomputed prototype

   - perform retrieval

     ```shell
     python retrieval.py \
     --test_dataset_path path-to-test-dataset \
     --dino_ckpt_path path-to-dino-checkpoint \
     --sam_ckpt_path path-to-sam-checkpoint \
     --prototype_path path-to-prototype \
     --mask_save_path path-to-save-mask
     ```

2. **Construct your own prototype for retrieval**

   - download the generated data

   - construct prototypes

     ```shell
     python gen_proto.py \
     --gen_data_path path-to-generated-data \
     --dino_ckpt_path path-to-dino-checkpoint \
     --prototype_save_path path-to-save-prototype
     ```

   - perform retrieval

## Acknowledgements

RA-COD builds upon a series of foundation models, including [DINOv2](https://github.com/facebookresearch/dinov2), [HQSAM](https://github.com/SysCV/sam-hq), [LLaVA](https://github.com/haotian-liu/LLaVA), and [Stable Diffusion](https://github.com/CompVis/stable-diffusion). Thanks for their excellent contributions.

# Contact

Feel free to contact me if there are any questions (email: duji@mail.nankai.edu.cn; WeChat: XH_duji).

## Citing

If you find our work interesting, please consider using the following BibTeX entry:

```latex
@ARTICLE{RA-COD,
  author={Du, Ji and and Wu, Jiesheng and Kong, Desheng and Hao, Fangwei and Xu, Jing and Li, Ping},
  journal={IEEE Transactions on Image Processing}, 
  title={RA-COD: Retrieval-Augmented Camouflaged Object Detection}, 
  year={2026}
  }

```

