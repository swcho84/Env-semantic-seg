# Env-semantic-seg
Semantic segmentation for Environment. It has different models and options to train over. 
(Ref.: forked from navblind/Sidewalk-semantic-seg, author: navblind)

## Working on..
- Using multiple GPUs
- Using OpenCV-DNN module for the inference

## Train From Scratch
Run this in the parent directory

```
python train.py --model=Model_name --num_epochs=20 --crop_height=128 --crop_width=192 --checkpoint_step=1 --dataset=CityScape
```
Training parameters
``` 
num_epochs= Number of epochs to train
crop_height= 128
crop_width= 192
checkpoint_step= Number of epochs to perform before saving latest checkpoint
dataset= Camvid, CityScape
batch_size = Batch size
model = FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, MobileUNet, MobileUNet-Skip, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet, FRRN-A, FRRN-B, PSPNet, GCN, DeepLabV3, DeepLabV3_plus, AdapNet, DenseASPP, DDSC,  BiSeNet, custom
```
Note: For model option custom, you have to code your own model. 

## Train from pretrained weights 

Run this in parent directory

```
python train.py --frontend=Model_name --num_epochs=10 --crop_height=128 --crop_width=192 --checkpoint_step=1 --dataset=CityScape
```

All parameters are the same except for frontend

```
frontend = ResNet50, ResNet101, ResNet152, MobileNetV2, InceptionV4, SEResNeXt50, SEResNeXt101
```

## Link to labeled dataset

Dataset options:
* Camvid (1000 images approx): https://drive.google.com/open?id=1oSeHb_7B8nuPdmTpUdiYZjFpD5b2Q7Zw
* CityScape (8000 images approx): https://drive.google.com/open?id=1mIuBYLQu80QUSRqLGaBEQh0OuYLafQ1r

**Note**: Put the dataset in the parent directory. It should look like this 

```
|__Parent Directory
  |__test
      |__Image1
      |__Image2
      .
      |__Image647
      
  |__test_labels
      |__Image1
      |__Image2
      .
      |__Image647
  |__train
      |__Image1
      |__Image2
      .
      |__Image2762
  |__train_labels
      |__Image1
      |__Image2
      .
      |__Image2762
  |__val
      |__Image1
      |__Image2
      .
      |__Image577
  |__val_labels
      |__Image1
      |__Image2
      .
      |__Image577
  |
  |__classes.csv
  
```
Note: Make sure classes.csv file exists in place. It is required for the program to run and has the description of the annotated labels. 

### Dataset attributes

Sidewalk color: rgb(0,0,192)
All other color: rgb(0,0,0)

## Predicting Res
```
python predict.py --crop_height=128 --crop_width=192 --model=ModelName --checkpoint_path=latest_checkpoint_path --image=TestImage.png
```







