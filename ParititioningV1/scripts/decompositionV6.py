#!/usr/bin/env python
import os, sys
import cv2
import numpy as np
import random
from Voronoi import Voronoi
from osgeo import gdal, gdalnumeric, ogr, osr
import threading
import math
from decomposition_tools import read_imagefile, get_center, get_points_circle, select_centroids, draw_voronoi_partition
from decomposition_tools import label_area, get_cells, new_voronoicentroids, max_min_area, name_C, name_Qmin, name_QP,  zigzag_path
from decomposition_tools import extract_max_area, draw_newPartition, set_level, get_centroid_max

#Partitions

def set_partition_level(n_uavs):
	if n_uavs <= 4: partition_level = 0
	elif n_uavs >4 and n_uavs<=7: partition_level = 1
	elif n_uavs >7 and n_uavs<=10: partition_level = 2
	elif n_uavs >10 and n_uavs<=13: partition_level = 3
	elif n_uavs >13 and n_uavs<=16: partition_level = 4

	return partition_level


def partition_base(filename, image, N, n_uavs, free_cells, c_level, radio, partition_level):
	rows, cols = image.shape[:2]
	#Get center of image
	center_x, center_y, p_corners_x, p_corners_y = get_center(image)
	#Get points[] in circle center in center_x, center_y 
	points = get_points_circle(image, center_x, center_y, p_corners_x, p_corners_y, radio)
	#Select centroids for voronoi partition
	centroids = select_centroids(N, center_x, center_y, points)
	#Draw voronoi partition
	white_img = draw_voronoi_partition(centroids, image, filename)
	#Labeled of free cells in each voronoi partition
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, centroids)
	min_area, max_area = max_min_area(labelarea)
	
	new_centroids = centroids
	print "Initial centroids:  \n" + str(centroids) + "\n"
	
	#Name agents type
	c_level = set_level(labelarea, c_level,0)
	print "c_level:  \n" +str(c_level) +"\n"

	if abs(n_uavs-3)<=1:
		
		labelarea, C_x, C_y = name_C(centroids,labelarea, final, rows, cols)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)
		new_centroids = new_voronoicentroids(labelarea, centroids, final)
		
		for c in centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea0.tif'), final)
		
		white_img = draw_voronoi_partition(new_centroids, image, filename)
		color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
		min_area, max_area = max_min_area(labelarea)
		
		c_level = set_level(labelarea, c_level,0)
		print "c_level:  " +str(c_level) 

		labelarea, C_x, C_y = name_C(new_centroids,labelarea, final, rows, cols)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)

		for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea1.tif'), final)
		#Get zizag path
		#path, path_end, uav_paths = zigzag_path(n_uavs, color, cells, label_cells, voro, filename)

	return final, white_img, max_area, labelarea, new_centroids, c_level


def partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, i):
	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, final)
	center_x, center_y, p_corners_x, p_corners_y = get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, p_corners_x, p_corners_y, radio)
	centroids = select_centroids(N, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	new_centroids.remove((mx,my))
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	#Name agents type
	c_level = set_level(labelarea, c_level, i)
	print "c_level:  " +str(c_level) +"\n"
	if abs(n_uavs -3)<=(3*i + 1):
		labelarea, C_x, C_y = name_C(new_centroids,labelarea, final, rows, cols)
		labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)
		new_centroids = new_voronoicentroids(labelarea, new_centroids, final)
		
		for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea0.tif'), final)
		
		white_img = draw_voronoi_partition(new_centroids, image, filename)
		color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
		min_area, max_area = max_min_area(labelarea)

		c_level = set_level(labelarea, c_level, 1)
		print "c_level:  " +str(c_level) +"\n"

		labelarea, C_x, C_y = name_C(new_centroids,labelarea, final, rows, cols)
		labelarea = name_Qmin(labelarea, C_x, C_y, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, C_x, C_y, cols, c_level, partition_level)

		for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea1.tif'), final)

	return final, white_img, max_area, labelarea, new_centroids, c_level



#Load 
filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_cropped.tif"

#Connected UAVs
n_uavs = 16
N = 0 # num of new partitions
partition_level = set_partition_level(n_uavs)
#Read file to cv2
image, gray_image, rows, cols = read_imagefile(filename)
raw_image = image.copy()
#Get size cells
cl = 50
rw = 40
#Get arrays
cells, cells_cost, free_cells = get_cells(cl, rw, gray_image)
#For each area center and partition_level
c_level = []
#For get points around a semicircle
radio = int(rows/10)

"""--------------------------------------BASE PARTITION------------------------------------------------"""
if partition_level >= 0:
	if n_uavs>4: N = 4
	else: N =n_uavs
	final, white_img, max_area, labelarea, new_centroids, c_level = partition_base(filename, image, N, n_uavs, free_cells, c_level, radio, partition_level)
	
"""----------------------------------------------------------------------------------------------------"""
"""--------------------------------------FIRST PARTITION-----------------------------------------------"""

if partition_level >= 1:
	if n_uavs>7: N = 4
	else: N =n_uavs -3

	final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 1)
	
"""-----------------------------------------------------------------------------------------------------"""
"""--------------------------------------SECOND PARTITION-----------------------------------------------"""
#Partitions for heuristic
if partition_level >= 2 :
	if n_uavs>10: N = 4
	else: N =n_uavs - 6
	final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 2)

"""-----------------------------------------------------------------------------------------------------"""
"""--------------------------------------THIRD PARTITION------------------------------------------------"""
#Partitions for heuristic
if partition_level >= 3:
	if n_uavs>13: N = 4
	else: N =n_uavs - 9
	final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 3)
"""-----------------------------------------------------------------------------------------------------"""
"""--------------------------------------FOURTH PARTITION-----------------------------------------------"""
#Partitions for heuristic
if partition_level == 4:
	N =n_uavs - 12
	final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 4)

	
"""-----------------------------------------------------------------------------------------------------"""
