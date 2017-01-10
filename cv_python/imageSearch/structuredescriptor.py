# -*- coding: utf-8 -*-
#__author__ = 'DragonFive'

import numpy as np
import cv2

class StructureDescriptor:
    __slot__ = ["dimension"]
    def __init__(self, dimension):
        self.dimension = dimension
    def describe(self, image):
        image = cv2.resize(image, self.dimension, interpolation=cv2.INTER_CUBIC)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image




