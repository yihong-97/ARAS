# ARAS Framework

## Prerequisites
The code is implemented with Python(3.6) and Pytorch(1.7).

## Datasets
- Download [GTA5 datasets](https://download.visinf.tu-darmstadt.de/data/from_games/)
- Download [Synthia datasets](http://synthia-dataset.net/download/808/)
- Download [Cityscapes datasets](https://www.cityscapes-dataset.com/)

## Training
### Pretrained Model
Pretrained models on source domian can be downloaded [here](https://drive.google.com/drive/folders/19Z-1bfvPsPKGyucnfM5E9HI9Tui03mY2?usp=sharing).

If you want to pretrain the model by yourself, you can refer to [here](https://github.com/ZJULearning/MaxSquareLoss).

### UDA
- GTA5-to-Cityscapes
```
## ResNet101-based
python train_UDA.py --source_dataset "gta5" --num_classes 19 --backbone "resnet101" --checkpoint_dir "./log/gta2city-res/" --pretrained_ckpt_file "../log/pretrainedmodles/gta5-res.pth"
```


## Testing
Our pretrained model is available [here](https://drive.google.com/drive/folders/1WGovcwmlunbL00RV-y5Aug_kWXZklQ4F?usp=sharing).
- GTA5-to-Cityscapes (example)
```
python evaluate.py --source_dataset "gta5" --num_classes 19 --backbone "resnet101" --split "test" --checkpoint_dir "./log/eval/gta2city-res-UDA/" --pretrained_ckpt_file "./log/gta2city-res/gta52cityscapesfinal.pth"
```



## Acknowledgments
This codebase is heavily borrowed from [UDAclustering](https://github.com/LTTM/UDAclustering) and [DAST_segmentation](https://github.com/yufei1900/DAST_segmentation).
