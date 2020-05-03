# YOLACT

An  adapted and simplified version of original YOLACT (https://github.com/dbolya/yolact/).

## Prepare

Original Yolact authors use pretrained weights of backbone 'resnet-101',which can be download from ['resnet101_reducedfc.pth'](https://drive.google.com/file/d/1vaDqYNB__jTB7_p9G6QTMvoMDlGkHzhP/view),

The file `'resnet101_reducedfc.pth'` should be put under `'./weights/'` directory.


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