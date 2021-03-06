# CIVIL-459: Spring 2022 Final Project

## Autors
Cengizhan Bektas (350828), Bastien Darbellay (288406), Zakarya Souid (287759)


## Milestone 1

To do object classification we decided to use ***YOLOv5***. YOLO belongs to the One-Stage Detectors family and is the ***state of the art*** algorithm for object detection due to its speed and accuracy. It divides images into a grid system where each cell in the grid is responsible for detecting objects within itself.

### 1) Collecting data

In order to train our custom model, we need to assemble a dataset of representative images with bounding box annotations around the objects that we want to detect. In addition we need our dataset to be in YOLOv5 format.

For this we used the open source model library [Roboflow](https://app.roboflow.com/private-qig8x/dlav-m1/5). We collected about 3.500 images and labeled all of them. To prevent ***overfitting*** and ***generalize*** the model well, we added some ***augmentation*** after resizing all images to 416x416:

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
!python train.py --img 416 --batch 128 --epochs 300 --data ~/data.yaml --weights yolov5s.pt --cache
```

Here, we are able to pass a number of arguments:

- img: define input image size
- ***batch***: determine batch size
- ***epochs***: define the number of training epochs
- data: location of the dataset
- ***weights***: specify a path to weights to start ***transfer learning*** from. Here we choose the generic COCO pretrained checkpoint.
- cache: cache images for faster training

#### 2.1) Model
There are provided several models with pretrained weights for specific environments. Since we will implement our algorithm on a robot we have chosen the model ***YOLOv5s*** which is recommended for mobile deployments. Larger models like YOLOv5x will produce better results, but have more parameters, require more CUDA memory to train, and are slower to run. We trained our model with default ***hyperparameters*** which are provided in *hyp.scratch.yaml*.

<img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png" width="650" align="center"/>  

<!-- #region -->
#### 2.2) Batch

Batch size is the number of data points used to train a model in each iteration. Choosing the right batch size is important to ensure convergence of the cost function and parameter values, and to the generalization of the model.

It determines the ***frequency of updates***. The smaller the batches, the more, and the quicker, the updates. The larger the batch size, the more accurate the gradient of the cost will be with respect to the parameters. That is, the direction of the update is most likely going down the local slope of the cost landscape. Having larger batch sizes, but not so large that they no longer fit in GPU memory, tends to improve parallelization efficiency and can accelerate training.

As we have been provided with the computing clusters at ***EPFL SCITAS*** with powerful GPU hardware, we used a large --batch-size to get better results. Small batch sizes produce poor batchnorm statistics and should be avoided.

#### 2.3) Insights

Here are the list of hardware components we were provided from the EPFL clusters. In addition, the first results from the first epochs are also showed:

<img src="./media/hardware_epfl_clusters.png" width="400" align="center"/> 
<br clear="left"/>
<img src="./media/training_epochs.png" width="400" align="center"/> 
<br clear="left"/>

The script to launch the jupyter notebook is as follows:

<img src="./media/launch_jupyter.png" width="400" align="center"/> 
<br clear="left"/>


#### 2.4) Monitor learning process
Training results and metrics are automatically logged and can be found in: ***/m1/yolov5/runs/train/exp/results.png***.

<img src="./m1/yolov5/runs/train/exp/results.png" width="750" align="center"/> 
<br clear="left"/>




**box_loss** represents how well the algorithm can locate the centre of an object and how well the predicted bounding box covers an object. 

**obj_loss** is essentially a measure of the probability that an object exists in a proposed region of interest. If the objectivity is high, this means that the image window is likely to contain an object. 

**cls_loss** gives an idea of how well the algorithm can predict the correct class of a given object.

For all the losses we can see, that these are descreasing with time. The initalization of the parameters are also really well, because we have an optimization directly after the first epoch. The step size of updating the parameters seems also be really good.

***Precision*** is the percentage of samples classified as positives that are truly positives. ***Recall*** is the percentage of positive samples that are correctly classified as positives. The ***mAP*** (mean Average Precision) is a metric to compare the performances of object detection systems. It is the average precision for each class in the data based on the model predictions. Average precision is related to the area under the precision-recall curve for a class. Taking the mean of these average individual-class-precision gives the mean Average Precision. 


#### 2.5) Training script and final model
The trained model can be found in: ***/m1/yolov5/runs/train/exp/weights/best.pt***.
The script we have used for train a custom model can be found in: ***/m1/dlav_yolov5_custom_training.ipynb***.

### 3) Measure performance of model

After training the model some inferences were made with the test data to evaluate the custom YOLOv5 detectors performance.

The results can be found in: ***/m1/yolov5/runs/detect/exp/***.

Since 5% of our whole data is to test the performance of the detector, there are more than 500 predictions. We just put some of these examples in the folder above to keep the size of the final submission as small as possible.

However, some examples are:

<img src="./m1/yolov5/runs/detect/exp/Photo-on-27-04-22-at-18_png.rf.425ab079ddcb34f75223380b5145705a.jpg" width="300" align="left"/> 
<img src="./m1/yolov5/runs/detect/exp/Photo-on-27-04-22-at-18--2-_png.rf.4e67994f24d827c8f11cdf409cd6a9eb.jpg" width="300" align="left"/>
<img src="./m1/yolov5/runs/detect/exp/Photo-on-27-04-22-at-18--1-_png.rf.52f6ecaae458715d406b3ddaa849c79f.jpg" width="300" align="left"/> 
<br clear="left"/>


### 4) Object detection

In order to do object detection from the output of the webcam, we implemented some functions.
The main idea is, to use the pretrained model from ultralytics to do object detection for each frame. As we are only interested in the person in the frame, we are filtering the pandas dataframe containing only the objects labeled as a person (see figure 1 below). After, we are using our own trained model to detect the gesture in order to detect the ***person of interest***. For this, we are searching for the person in the list of detected persons with the ***biggest interception*** of the rectangles between each person and the rectangle of the gesture. Then we are selecting the person with the biggest overlap (see figure 2 below). To ***avoid unneeded processing***, we are removing all persons in the list which has a smaller area of a specific ***threshold*** we set, to guarantee that we are only interested into the persons which are in the ***foreground***. Finally we are calculating the coordinates of the ***bounding box*** to return the position of the rectangle of the person of interest.

<img src="./media/list_persons.png" width="400" align="left"/> 
<img src="./media/list_person_of_interest.png" width="400" align="left"/> 
<br clear="left"/>

The script can be found in: ***/m1/m1.ipynb***.

### 5) Problems

In the beginning we had some problems to collect data in order to train our custom model for the detection of the gesture. Finally we were able to find a huge set of data which were also allocated with the bounding box around the object we want to detect. In addition we initially had problems to train the model due the limited computational power on google colab. For this, we spend a lot of time to set up the environment on SCITAS to train our model without hardware limitations. Because of the fact that we added some augmentation to the data, we didnt have any problems to detect the person of interest in different situations (illuminated room, white background, ...).

Another issue we had, appeared when two persons were one behind the other. The overlapping area of the hand and the persons were equal and could lead to a wrong initialization of the person of interest. To solve this problem, we decided to identify the person of interest not only based on the overlapping area with the hand, but also based on the size of the bounding box of the person. We assumed that the person of interest would be close to the robot and therefore have a bigger bounding box than the persons in the back. With this modification, we were able to accurately detect the person of interest, even in ambiguous conditions.

### 6) Code description

object_detect: detects objects using yolov5 model

hand_detect: detects objects using hand model

get_persons: takes all object detected and return objects that are person

get_personOfInterest: match the detected hand object and try to match the bounding box with the bounding boxes of all the person objects

For detection, JS response is first converted to OpenCV Image, then object_detect and hand_detect is ran to detect the objects. The person of interest is then identified and bounding boxes are finally drawn around the person we detects.

### 7) Run the code

1) Upload the folder "m1" to your Google Drive
2) Upload the file best.pt to your Google Colab folder on your drive ("drive/MyDrive/Colab Notebooks/")
3) Open m1.ipynb in Google Colab and run each cell one after the other. At the end you should see the feed of your front camera appear.

To change the camera from "front" to "rear", change the facingMode in the forth cell from "user" to "environment". 

## Milestone 2

We initially planned on trying both DeepSort and SiamFC to acheive Milestone 2. 

#### Andrew please talk about deepsort quickly and why we didnt choose it 


SiamFC is a basic tracking algorithms with a fully-convolutional Siamese network. In this approach, a deep conv-net is trained to address a more general similarity learning problem in an initial offline phase, and the coresponding function is then evaluated online during tracking. The Siamese network is trained to located an exemplar image within a larger search image, where the similarity learning is done within the bilinear layers.
We started by implementing SiamFC. 

Siamese Network based tracking algorithms are Single Object Trackers (SOT). We choose to track with SOT because of our needs in this application. We also preferred to not use detection based tracking, which is really common in tracking, because we thought this approach to be more interesting, given we already had to do detection for milestone 1.
A demo version can be seen in the Milestone2demo1.ipynb Jupyter notebook.

We also tried to use SiamMask, a more modern Siamese network implementation. We planned on using it for the race and managed to make it work in Colab. The Milestone2demo2.ipynb can demo it.

### SiamFC

#### Principle

We would like to thank Huang Lianghua for his Pytorch implementation of SiamFC  
https://github.com/huanglianghua/siamfc-pytorch  

and Luca Bertinetto , Jack Valmadre , João F. Henriques, Andrea Vedaldi and Philip H.S. Torr at Oxford University for the *Fully-Convolutional Siamese Networks for Object Tracking* publication   
https://www.robots.ox.ac.uk/~luca/siamese-fc.html  

we used is a Siam-FC model with an AlexNet backend. The model takes in an image where the feature (here a person) is and its location on that image, and a search image, and uses the backend to process each image into an embedding, then uses cross-correlation to find the search image in the instance image.  

Siamese networks differ from most neural networks in that instead of classifying the inputs, they differentiate between two inputs. Siamese networks are "Siamese" because they consist of two neural networks that are from the same class and have the same weights. Once the two images go through the two networks, it is then fed to a cross entropy loss function that calculates the similarity of two inputs, optimizing the networks.  

 
<img src="./m2/images/SiamFCprinciple.png" width="800" align="left"/>    <br />

<img src="./m2/images/ConvMap.png" width="800" align="left"/>    <br />



#### Network

The above image illustrates the SiamFC architecture. A trained image, z, and a search image, x, are passed into identical neural networks. The images are then rescaled and passed into a cross correlation function that generates a score map of the search image and the predictions of the tracked object within the image.

The tracker uses the AlexNet architecture, with 5 Convplutional layers and 3 Fully Connected layers.
The Convolutional Layers extract the object that corresponds to the unique ID within a given image passed through the neural network. The Fully Connected Layers use Max Pooling with 3x3 pooling windows with a stride of 2 between the adjacent windows in order to downsample the width and height of the tensors and keep the depth the same between windows. In addition, the Convolutional Layers each use a Rectified Linear Unit (ReLU) Activation function that clamp down any negative values in the network to 0, with positive values unchanged. The result of this function serves as the output of the Convolutional Layer.

A loss function used in this model is a cross entropy loss function. The loss is first computed by taking in two parameters of the same class, the first parameter being 1 or 0 depending on the true value of the model and the second being model's prediction for that class. The loss function then returns the mean logarithm of the negative product between the two parameters and the training code stores the weights at which at the minimum loss occurs.

#### Implementation 

All the implementation we found of SiamFC were not real time and only analyzed a pre-recorded video. 
We slightly modified the SiamFC-pytorch library and wrote a notebook to use it in Google Colab in real time. 


As soon as a person of interest is detected, through milestone 1, their bounding box is given to the initialization part of the tracker with the image it was detected on. After that, each frame is given to the tracker through an updating function, without redoing the initialization again under any circumstance. If the tracker looses the person of interest, it simple catches it back when their in the frame again, without asking to be reinitialized. 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zak-souid/DLAV_project/blob/main/m2/Milestone2demo1.ipynb)


### SiamMask

#### Principle

We would like to thank Qiang Wang for his pytorch implementation of SiamMask  
https://github.com/foolwood/SiamMask  

and Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, and Philip H.S. Torr at Oxford University for the *Fast Online Object Tracking and Segmentation: A Unifying Approach* publication  
https://www.robots.ox.ac.uk/~qwang/SiamMask/

Our intention was to use SiamMask, as it improves on what we liked about SiamFC and added some features we found useful, as the confidence metric and the mask feature. 

SiamMask works largely as SiamFC but adds a by augmenting the loss with a binary segmentation task.  


#### Network 

SiamMask uses a ResNet-50 Network until the final convolutional layer of the 4-th stage. In order to obtain a high spatial resolution in deeper layers,  the output stride is reduced to 8 by using convolutions with stride 1. Moreover, we increase the receptive field by using dilated convolutions [6]. In our model, we add to the shared backbone fθ an unshared adjust layer (1×1 conv with 256 outputs).

<img src="./m2/images/siammask.png" width="800" align="left"/>  <br />   


#### Implementation 

Again, the implementations of SiamMask were not real time. We adapted it to work with the webcam in colab.  
Our implementation uses the confidence metric and make it visible in the webcam box : when the tracker is sure of seeing the tracked individual, the webcam box turns black except the individual. When confidence lowers, the opacity of the black lowers too and we see what the webcam sees again. This is very useful in case the robot looses the tracked person, so it does not just go to a random position. Also, with bounding "blobs" instead of bounding boxes, we can make sure the middle of the box does not point to somewhere that is not the person of interest.  
You can try it in this colab notebook : 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zak-souid/DLAV_project/blob/main/m2/Milestone2demo2.ipynb)


## Milestone 3

JUST EXPLAIN WE JUST COMBINED M1 AND M2 AND DID SOME DEBUG TO CODE ETC
<!-- #endregion -->

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

