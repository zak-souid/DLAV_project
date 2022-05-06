# CIVIL-459: Spring 2022 Final Project

## Autors
Cengizhan Bektas (350828)


## Milestone 1

### 1) Collecting data

In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. In addition we need our dataset to be in YOLOv5 format.

For this we used the open source model library [Roboflow](https://app.roboflow.com/private-qig8x/dlav-m1/5). We collected about 3500 images and labeled all of them. To prevent overfitting and generalize the model well, we added some augmentation after resizing all images to 416x416:

- Rotation between -10° and +10°
- Saturation between -10% and +10%
- Brightness between -15% and +15%
- Noise up to 4% of pixels

After adding some augmentation the amount of our images was multiplied by 3 and finally we got more then 10.000 images to train our model.

### 2) Train our custom model

For best training results we have chosen [YOLOv5](https://github.com/ultralytics/yolov5) as object detection architecture with pretrained models on the COCO dataset. In the GitHub project from [Ultralytics](https://ultralytics.com/) we ran a script to train our own model:
```python
!python train.py --img 416 --batch 16 --epochs 150 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

Here, we are able to pass a number of arguments:

- img: define input image size
- **batch**: determine batch size
- **epochs**: define the number of training epochs
- data: location of the dataset
- **weights**: specify a path to weights to start **transfer learning** from. Here we choose the generic COCO pretrained checkpoint.
- cache: cache images for faster training

#### Model
There are provided several models with pretrained weights for specific environments. Since we will implement our algorithm on a robot we have chosen the model **YOLOv5s** which is recommended for mobile deployments. Larger models like YOLOv5x will produce better results, but have more parameters, require more CUDA memory to train, and are slower to run. We trained our model with default **hyperparameters** which are provided in *hyp.scratch.yaml*.

<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png" width="650" align="center"/>  


#### Batch

As we have been provided with the computing clusters at EPFL **SCITAS** with powerful GPU hardware, we used a large *--batch-size* to get better results. Small batch sizes produce poor batchnorm statistics and should be avoided.

#### Monitor learning process
1) increasing accuracy
2) Loss curve

### 3) Object detection

## Milestone 1
pdjsoüfjnsd+ofnsüodfn


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

