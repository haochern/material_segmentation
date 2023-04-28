# material_segmentation

## Installation

```bash
git clone --recurse-submodules https://github.com/haochern/material_segmentation.git  # clone recursively
```
For install submodules, please refer to the guidelines in [SegFormer](https://github.com/NVlabs/SegFormer) and [segment-anything](https://github.com/facebookresearch/segment-anything).

## Dataset
[The Dense Material Segmentation Dataset](https://github.com/apple/ml-dms-dataset)

## Train
```bash
python3 SegFormer/tools/train.py SegFormer/local_configs/segformer/B1/segformer.b1.512x512.dms.160k.py
```

## Evaluation

Download trained SegFormer weights
([google drive](https://drive.google.com/file/d/1HDLwNOj5w2ML-8KWyc61Mu5TsHbULT-1/view?usp=sharing))

Run Demo
```bash
python3 image_demo.py <image.png> <segformer_config.py> <segformer.pth> <sam_vit_h.pth>
```
Preprocess RGBD data
```bash
python3 rgbd_preprocess.py <data_dir> <segformer_config.py> <segformer.pth> <sam_vit_h.pth>
```
Visualize <data.npy> file
```bash
python3 npy_reader.py <data_dir> <segformer_config.py> <segformer.pth>
```