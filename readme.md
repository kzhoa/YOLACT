# YOLACT

An  adapted and simplified version of original YOLACT (https://github.com/dbolya/yolact/).

## Prepare

Original Yolact authors use pretrained weights of 'resnet-101',which can be download from 'resnet101_reducedfc.pth'(),

The file 'resnet101_reducedfc.pth' should be put under './weights/' directory.


## Train

to train a new model with default save folder './weights/'
```bash
python train.py 
```

or indicate one
```bash
python train.py  --save_folder weights/
```

to train with a weight file 
```bash
python train.py --resume weights/yolact_base_0_30000.pth
```