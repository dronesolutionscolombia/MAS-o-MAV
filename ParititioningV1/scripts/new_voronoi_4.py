#!/usr/bin/env python
import os, sys
import cv2
import numpy as np
import random
from Voronoi import Voronoi
from osgeo import gdal, gdalnumeric, ogr, osr
import threading
import math
from Tools import read_imagefile, get_center, get_points_circle, select_centroids, draw_voronoi_partition
from Tools import label_area, get_cells, new_voronoicentroids, max_min_area, name_C, name_Qmin, name_QP,  zigzag_path
from Tools import extract_max_area, draw_newPartition, set_level
#Load 
filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_cropped.tif"

#Connected UAVs
n_uavs = 16
partition_level = 0
n_partitions = 0
N = 0
#Partitions
if n_uavs <= 4: partition_level = 0
elif n_uavs >4 and n_uavs<=7: partition_level = 1
elif n_uavs >7 and n_uavs<=10: partition_level = 2
elif n_uavs >10 and n_uavs<=13: partition_level = 3
elif n_uavs >13 and n_uavs<=16: partition_level = 4

#Read file to cv2
image, gray_image, rows, cols = read_imagefile(filename)
print (rows, cols)
raw_image = image.copy()
cl = 50
rw = 40
cells, cells_cost, free_cells = get_cells(cl, rw, gray_image)
c_level = []
"""--------------------------------------BASE PARTITION------------------------------------------------"""
if partition_level >= 0:
	if n_uavs>4: N = 4
	else: N =n_uavs

	#Get center of image
	center_x, center_y, p_corners_x, p_corners_y = get_center(image)
	#Radio for partition (?)
	radio = int(rows/10)
	#Get points[] in circle center in center_x, center_y 
	points = get_points_circle(image, center_x, center_y, p_corners_x, p_corners_y, radio)
	#Select centroids for voronoi partition
	centroids = select_centroids(N, center_x, center_y, points)
	#Draw voronoi partition
	white_img = draw_voronoi_partition(centroids, image, filename)
	#Labeled of free cells in each voronoi partition
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, centroids)

	#Adjust the centroids around minarea 
	delta_x = 10
	delta_y = 10
	new_centroids = new_voronoicentroids(labelarea, delta_x , delta_y)

	#Draw voronoi partition
	white_img = draw_voronoi_partition(new_centroids, raw_image, filename)
	#Labeled of free cells in each voronoi partition
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	#Name agents type
	
	c_level = set_level(labelarea, c_level,0)
	print "c_level:  " +str(c_level) +"\n"
	if n_uavs<=4:
		labelarea, C_x, C_y = name_C(labelarea, rows, cols)
		#labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level)

	#Get zizag path
	#path, path_end, uav_paths = zigzag_path(n_uavs, color, cells, label_cells, voro, filename)
"""----------------------------------------------------------------------------------------------------"""
"""--------------------------------------FIRST PARTITION-----------------------------------------------"""
#Partitions for heuristic
if partition_level >= 1:
	if n_uavs>7: N = 4
	else: N =n_uavs -3
	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	center_x, center_y, p_corners_x, p_corners_y = get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, p_corners_x, p_corners_y, radio)
	centroids = select_centroids(N, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	#Name agents type
	c_level = set_level(labelarea, c_level, 1)
	print "c_level:  " +str(c_level) +"\n"
	if n_uavs<=7:
		labelarea, C_x, C_y = name_C(labelarea, rows, cols)
		labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)

	
"""-----------------------------------------------------------------------------------------------------"""
"""--------------------------------------SECOND PARTITION-----------------------------------------------"""
#Partitions for heuristic
if partition_level >= 2 :
	if n_uavs>10: N = 4
	else: N =n_uavs - 6
	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	center_x, center_y, p_corners_x, p_corners_y = get_center(maxarea_img)
	points = get_points_circle(  maxarea_img,center_x, center_y, p_corners_x, p_corners_y, radio)
	centroids = select_centroids(N, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)

	#Name agents type
	c_level = set_level(labelarea, c_level, 2)
	print "c_level:  " +str(c_level) +"\n"

	if n_uavs<=10:
		labelarea, C_x, C_y = name_C(labelarea, rows, cols)
		labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level,partition_level)


"""-----------------------------------------------------------------------------------------------------"""
"""--------------------------------------THIRD PARTITION------------------------------------------------"""
#Partitions for heuristic
if partition_level >= 3:
	if n_uavs>13: N = 4
	else: N =n_uavs - 9
	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	center_x, center_y, p_corners_x, p_corners_y = get_center(maxarea_img)
	points = get_points_circle( maxarea_img,center_x, center_y, p_corners_x, p_corners_y, radio)
	centroids = select_centroids(N, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)

	c_level = set_level(labelarea, c_level, 3)
	print "c_level:  " +str(c_level) +"\n"
	if n_uavs<=13:
		labelarea, C_x, C_y = name_C(labelarea, rows, cols)
		labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)
"""-----------------------------------------------------------------------------------------------------"""
"""--------------------------------------FOURTH PARTITION-----------------------------------------------"""
#Partitions for heuristic
if partition_level == 4:
	N =n_uavs - 12
	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	center_x, center_y, p_corners_x, p_corners_y = get_center(maxarea_img)
	points = get_points_circle( maxarea_img,center_x, center_y, p_corners_x, p_corners_y, radio)
	centroids = select_centroids(N, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	c_level = set_level(labelarea, c_level, 4)
	print "c_level:  " +str(c_level) +"\n"
	labelarea, C_x, C_y = name_C(labelarea, rows, cols)
	labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
	labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)
	
"""-----------------------------------------------------------------------------------------------------"""
