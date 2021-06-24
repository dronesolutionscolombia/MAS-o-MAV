#!/usr/bin/env python
import os, sys
import cv2
import numpy as np
import random
from Voronoi import Voronoi
from osgeo import gdal, gdalnumeric, ogr, osr
import threading
import math
from decom_tools import *

		
#Load 
filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_cropped.tif"

#Connected UAVs
n_uavs = 5
#Get size cells - this is from sensor characteristics
cl = 50
rw = 40
decom_level = 0

#decomposition(n_uavs, filename, cl, rw, decom_level)
#Read file to cv2
image, gray_image, rows, cols = read_imagefile(filename)
#Get arrays
cells, cells_cost, free_cells = get_cells(cl, rw, gray_image)
#For get points around a semicircle
radio = int(rows/10)

decomposition (n_uavs, image, filename, cl, rw, 0)

#new_captains(filename, image, radio, free_cells, 2, cl, rw)