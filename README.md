# CIVIL-459: Spring 2022 Final Project

## Autors
Cengizhan Bektas (350828)


## Milestone 1

To do object classification we decided to use YOLOv5. YOLO belongs to the One-Stage Detectors family and is the state of the art algorithm for object detection due to its speed and accuracy. It divides images into a grid system where each cell in the grid is responsible for detecting objects within itself.

### 1) Collecting data

In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. In addition we need our dataset to be in YOLOv5 format.

For this we used the open source model library [Roboflow](https://app.roboflow.com/private-qig8x/dlav-m1/5). We collected about 3.500 images and labeled all of them. To prevent overfitting and generalize the model well, we added some augmentation after resizing all images to 416x416:

- Rotation between -10° and +10°
- Saturation between -10% and +10%
- Brightness between -15% and +15%
- Noise up to 4% of pixels

After adding some augmentation the amount of our images was multiplied by 3 and finally we got more then 10.000 images to train our model.

Split of the data is as follows:

- 80% training
- 15% validation
- 5% test


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

Batch size is the number of data points used to train a model in each iteration. Choosing the right batch size is important to ensure convergence of the cost function and parameter values, and to the generalization of the model.

It determines the frequency of updates. The smaller the batches, the more, and the quicker, the updates. The larger the batch size, the more accurate the gradient of the cost will be with respect to the parameters. That is, the direction of the update is most likely going down the local slope of the cost landscape. Having larger batch sizes, but not so large that they no longer fit in GPU memory, tends to improve parallelization efficiency and can accelerate training.

As we have been provided with the computing clusters at EPFL SCITAS with powerful GPU hardware, we used a large --batch-size to get better results. Small batch sizes produce poor batchnorm statistics and should be avoided.

#### Monitor learning process
Training results and metrics are automatically logged and can be found in: ***/dlav_yolov5_custom_training.ipynb***.

### 3) Measure performance of model

After training the model some inferences were made with the test data to evaluate the custom YOLOv5 detectors performance.

The results can be found in: ***/dlav_yolov5_custom_training.ipynb***.

### 4) Object detection

HERE YOU CAN DESCRIBE THE FUNCTIONS REALLY SHORTLY WHICH WERE IMPLEMENTED TO DO THE DETECTION... SEE CODE ON GITHUB

object_detect: detects objects using yolov5 model

hand_detect: detects objects using hand model

get_persons: takes all object detected and return objects that are person

get_personOfInterest: match the detected hand object and try to match the bounding box with the bounding boxes of all the person objects

For detection, JS response is first converted to OpenCV Image, then object_detect and hand_detect is ran to detect the objects. The person of interest is then identified and bounding boxes are finally drawn around the person we detects.

## Milestone 2

SIAMFC is a basic tracking algorithms with a fully-convolutional Siamese network. In this approach, a deep conv-net is trained to
address a more general similarity learning problem in an initial offline phase, and the coresponding function is then evaluated online during tracking. The Siamese network is trained to located an exemplar image within a larger search image, where the similarity learning is done within the bilinear layers.

EXPLAIN LOGIC WE IMPLEMENTED HOW TO TO THE DETECTION

EXPLAIN KEY DECISIONS (HYPERPARAMETERS WE SET )

## Milestone 3

JUST EXPLAIN WE JUST COMBINED M1 AND M2 AND DID SOME DEBUG TO CODE ETC


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

