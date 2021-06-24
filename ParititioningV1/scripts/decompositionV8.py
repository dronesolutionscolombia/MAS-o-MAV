#!/usr/bin/env python
import os, sys
import cv2
import numpy as np
import random
from Voronoi import Voronoi
from osgeo import gdal, gdalnumeric, ogr, osr
import threading
import math
from decom_tools_2 import *

		
#Load
filename = "/home/liseth/catkin_ws/maps/cerritos/cerritos_modified_cropped.tif"

#Connected UAVs
n_uavs = 5
#Get size cells - this is from sensor characteristics
cl = 35
rw = 30

decom_level = 0

#Read file to cv2
image, gray_image = read_imagefile(filename)


decomposition(n_uavs, image, filename, cl, rw, decom_level)

#new_captains(filename, image, radio, free_cells, 2, cl, rw)