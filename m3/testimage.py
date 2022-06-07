import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn.functional as F


import cv2
import numpy as np
import PIL
import time
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from siamfc import ops
from siamfc.backbones import AlexNetV1
from siamfc.heads import SiamFC
from siamfc.losses import BalancedLoss
from siamfc.datasets import Pair
from siamfc.transforms import SiamFCTransforms
from siamfc.siamfc import Net
from siamfc.siamfc import TrackerSiamFC


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, n_c):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.box = torch.nn.Linear(n_hidden, n_output-1)   # output layer
        self.logit = torch.nn.Linear(n_hidden, 1)

        self.conv1 = torch.nn.Sequential(         # input shape (3, 80, 60)
            torch.nn.Conv2d(
                in_channels = n_c,            # input height
                out_channels = 8,             # n_filters
                kernel_size = 5,              # filter size
                stride = 2,                   # filter movement/step
                padding = 0,
            ),
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(kernel_size = 2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 8,
                            out_channels = 16,
                            kernel_size = 5,
                            stride = 2,
                            padding = 0),
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 16,
                            out_channels = 8,
                            kernel_size = 1,
                            stride = 1,
                            padding = 0),
            torch.nn.ReLU(),                      # activation
            #torch.nn.MaxPool2d(2),
        )
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = feat.view(feat.size(0), -1)
        x2 = F.relu(self.hidden(feat))      # activation function for hidden layer

        out_box = F.relu(self.box(x2))            # linear output
        out_logit = torch.sigmoid(self.logit(x2))

        return out_box, out_logit

class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        super(Detector, self).__init__()

        # init variables
        self.init = False
        self.persOfInterest = False
        self.bb_persOfInterest = [0,0,0,0]
        self.img_0 = np.zeros([480,640,3])
        self.last_img = np.zeros([480,640,3])
        self.net_path = 'siamfc_alexnet_e50.pth'
        self.tracker = TrackerSiamFC(net_path=self.net_path)
        self.bbox = ''

        # Model
        self.obj_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.hand_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./data/best.pt')



    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def obj_detect(img):
        """
        Params:
            img: OpenCV BGR image
        Returns:
            res: pandas dataframe containing all objects detected by the model
        """
        results = obj_model(img)
        res = results.pandas().xyxy[0]  # img1 predictions (pandas)
        return res

    def get_persons(objects):
        """
        Params:
            objects: pandas dataframe containing different objects
        Returns:
            persons: pandas dataframe containing only the objects labeled as person
        """
        persons = objects.loc[objects['name'] == "person"]
        idx_del = []
        areas = []

        for idx, pers in persons.iterrows():
            dx = pers["xmax"]-pers["xmin"]
            dy = pers["ymax"]-pers["ymin"]
            areas.append(dx*dy)
            if(areas[-1] < 10000):
                idx_del.append(idx)
        persons = persons.drop(idx_del)
        persons = persons.reset_index()


        return persons, areas

    def get_personOfInterest(persons, hand, persons_area):  # returns None if rectangles don't intersect
        """
        Params:
            persons: pandas dataframe containing only objects labeled as person
            hand: pandas dataframe containing only the detected hands (labeled as 5)
        Returns:
            idx_max: integer corresponding to the index in the persons dataframe that corresponds to the person of interest
                    is equal to None if no hand is being detected or the hand does not belongs to any of the detectet persons
        """
        margin = 50 # For numerical inaccuracies
        idx_max = 0
        area_max = 0
        a_min = [-1,-1]
        a_max = [-1,-1]
        if not hand.empty:
            for idx, pers in persons.iterrows():
                dx = np.minimum(pers["xmax"], hand["xmax"].iloc[0]) - np.maximum(pers["xmin"], hand["xmin"].iloc[0])
                dy = np.minimum(pers["ymax"], hand["ymax"].iloc[0]) - np.maximum(pers["ymin"], hand["ymin"].iloc[0])
                if(dx<0 and dy<0):
                    area = -dx*dy
                else:
                    area = dx*dy
                if (area >= area_max - margin and persons_area[idx] >= persons_area[idx_max]):
                    area_max = area
                    idx_max = idx
                    # actual overlapping area
                    a_min = [int(np.maximum(pers["xmin"], hand["xmin"].iloc[0])),int(np.maximum(pers["ymin"], hand["ymin"].iloc[0]))]
                    a_max = [int(np.minimum(pers["xmax"], hand["xmax"].iloc[0])),int(np.minimum(pers["ymax"], hand["ymax"].iloc[0]))]

        if (area_max>0):
            return idx_max, a_min, a_max
        else:
            return None, a_min, a_max

    def hand_detect(self, img):
        """
        Params:
            img: OpenCV BGR image
        Returns:
            res: pandas dataframe containing all objects detected by the model
        """
        results = self.hand_model(img)
        res = results.pandas().xyxy[0]  # img1 predictions (pandas)
        return res

    def detect(self, img):
        # detect all objects
        obj = self.obj_detect(img)
        # keep only the persons
        pers, persons_area = self.get_persons(obj)
        # detect hand
        hand = self.hand_detect(img)

        # identify person of interest
        idx_max, area_min, area_max = self.get_personOfInterest(pers,hand, persons_area)

        # create transparent overlay for bounding box
        bbox_array = np.zeros([480,640,4], dtype=np.uint8)

        # draw bounding box of the hand (if multiple hands, just the first one)
        if (not hand.empty) and area_min!=None:
            bbox_array = cv2.rectangle(bbox_array,(int(hand["xmin"].iloc[0]),int(hand["ymin"].iloc[0])), (int(hand["xmax"].iloc[0]),int(hand["ymax"].iloc[0])), (0, 0, 255), 2)
            bbox_array = cv2.rectangle(bbox_array,tuple(area_min), tuple(area_max), (255, 0, 255), 2)

        # draw bounding boxes on overlay
        for index, row in pers.iterrows():
            start_point = (int(row["xmin"]), int(row["ymin"]))
            end_point = (int(row["xmax"]), int(row["ymax"]))
            if (index == idx_max):
                color = (0, 255, 0)
                self.persOfInterest = True
                self.bb_persOfInterest = [start_point[0], start_point[1], end_point[0]-start_point[0], end_point[1]-start_point[1]]
                #print(bb_persOfInterest)
            else:
                color = (255, 0, 0)
            thickness = 2
            # bbox_array = cv2.rectangle(bbox_array,start_point, end_point, color, thickness)
            bbox_array = [(int(row["xmin"])+int(row["xmax"]))/2, (int(row["ymin"])+int(row["ymax"]))/2,(int(row["xmax"])-int(row["xmin"])),(int(row["ymax"])-int(row["ymin"]))]

        return bbox_array

    def tracking(self, img):

        if not self.init:
            self.img_0 = img
            self.last_img = img
            self.init = True

        # convert JS response to OpenCV Image
        pair_img = [self.img_0, self.last_img, img]

        boxes, times = self.tracker.track(pair_img, self.bb_persOfInterest, visualize=False)
        #print(boxes[1])
        start_point = (int(boxes[-1][0]),int(boxes[-1][1]))
        #print(start_point)
        end_point = (int(boxes[-1][0])+int(boxes[-1][2]),int(boxes[-1][1])+int(boxes[-1][3]))
        #print(end_point)

        # bbox_array = np.zeros([480,640,4], dtype=np.uint8)
        # bbox_array = cv2.rectangle(bbox_array, start_point, end_point, (255, 255, 0), 2)
        self.last_img=img
        bbox_array = [int(boxes[-1][0])+int(boxes[-1][2])/2,int(boxes[-1][1])+int(boxes[-1][3])/2,int(boxes[-1][2]),int(boxes[-1][3])]
        return bbox_array

    def forward(self, img):
        ##Add a dimension
        img = np.expand_dims(img.transpose(1,0,2), 0) / 255

        ch1 = img[:,:,:,0].copy()
        ch3 = img[:,:,:,2].copy()

        img[:,:,:,0] = ch3
        img[:,:,:,2] = ch1

        # print(img.shape)
        # print(img)

        ##Preprocess
        img = (img - self.mean)/self.std

        ##Transpose to model format
        if(img.shape[1] != self.num_channels):
            img = img.transpose((0,3,1,2))

        if not self.persOfInterest:
            bbox_array = self.detect(img)

        else:
            bbox_array = self.tracking(img)


        # print(img.shape)
        # print(img)

        ##Detect
        # with torch.no_grad():
        #     pred_y_box, pred_y_logit = self.model.forward(torch.tensor(img, dtype=torch.float32))

        #     pred_y_box, pred_y_logit = pred_y_box.numpy(), pred_y_logit.numpy()
        #     pred_y_label = pred_y_logit > 0.5
        #     pred_bboxes = pred_y_box * self.img_size
        #     # pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)

        return pred_bboxes[0], pred_y_label[0]
