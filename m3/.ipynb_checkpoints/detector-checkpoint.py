# import modules
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import PIL
from PIL import Image

import os
import cv2
import time
import pandas as pd
import numpy as np
from collections import namedtuple
from got10k.trackers import Tracker

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

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
        self.hand_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./../best.pt')

    def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()

    def obj_detect(self, img):
        """
        Params:
            img: OpenCV BGR image
        Returns:
            res: pandas dataframe containing all objects detected by the model
        """
        results = self.obj_model(img)
        res = results.pandas().xyxy[0]  # img1 predictions (pandas)
        return res

    def get_persons(self, objects):
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
            if(areas[-1] < 6000):
                idx_del.append(idx)
        persons = persons.drop(idx_del)
        persons = persons.reset_index()


        return persons, areas

    def get_personOfInterest(self, persons, hand, persons_area):  # returns None if rectangles don't intersect
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
        bbox_array = [0,0,0,0]
        label = [0]

        # draw bounding boxes on overlay
        for index, row in pers.iterrows():
            start_point = (int(row["xmin"]), int(row["ymin"]))
            end_point = (int(row["xmax"]), int(row["ymax"]))
            if (index == idx_max):
                color = (0, 255, 0)
                self.persOfInterest = True
                self.bb_persOfInterest = [start_point[0], start_point[1], end_point[0]-start_point[0], end_point[1]-start_point[1]]
            else:
                color = (255, 0, 0)
            thickness = 2

        if(idx_max is None):
            bbox_array = [0,0,0,0]
            label = [0]

        return bbox_array, label

    def tracking(self, img):

        if not self.init:
            self.img_0 = img
            self.last_img = img
            self.init = True

        # convert JS response to OpenCV Image
        pair_img = [self.img_0, self.last_img, img]
        boxes, times = self.tracker.track(pair_img, self.bb_persOfInterest, visualize=False)
        # start point
        start_point = (int(boxes[-1][0]),int(boxes[-1][1]))
        # end point
        end_point = (int(boxes[-1][0])+int(boxes[-1][2]),int(boxes[-1][1])+int(boxes[-1][3]))
        self.last_img=img
        bbox_array = [float(int(boxes[-1][0])+int(boxes[-1][2])/2),float(int(boxes[-1][1])+int(boxes[-1][3])/2),float(boxes[-1][2]),float(boxes[-1][3])]
        return bbox_array

    def forward(self, img):
        img = np.array(img)
        label = [0]
        if not self.persOfInterest:
            print("----Detect----")
            pred_bboxes, label = self.detect(img)
        else:
            print("----Tracking----")
            pred_bboxes = self.tracking(img)
            label = [1]

        return pred_bboxes, label
