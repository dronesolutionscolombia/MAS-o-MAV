#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from copy import deepcopy
sys.path.insert(0, "/usr/local/lib/python2.7/site-packages")
import pandas as pd
from matplotlib import pyplot as plt
import random

import cv2
import numpy as np
import random
from osgeo import gdal, gdalnumeric, ogr, osr
import threading
import time


filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_cropped.tif"
n_uavs = 15

start_time = time.time()
_image = cv2.imread(filename)
gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
rows, cols = _image.shape[:2]

gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
new = 255 - gray
gray = cv2.GaussianBlur(gray, (3, 3), 3)
t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
ff, contours, dd  = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final = np.zeros(_image.shape,np.uint8)
mask = np.zeros(gray.shape,np.uint8)

i=0
color_ = []
p_corners_x=[]
p_corners_y=[]
l_x = 0
r_x = 0
u_y = 0
b_y = 0


for c in contours:
	area = cv2.contourArea(c)
	
	if area > 500 and area<(rows*cols)/2:
		#cv2.drawContours(img, [c], 0, (0, 255, 0), 1, cv2.LINE_AA)
		color_.append((random.randint(10,240), random.randint(10,240), random.randint(10,240)))
		cv2.drawContours(mask, contours, i, 255, -1)
		#cv2.drawContours(final, contours, i, cv2.mean(src, mask), -1)
		cv2.drawContours(final, contours, i, color_[i], -1)
		
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		lx, ly = extLeft
		rx, ry = extRight
		ux, uy = extTop
		bx, by = extBot
		
		p_corners_x.append(lx)
		p_corners_x.append(rx)
		p_corners_y.append(uy)
		p_corners_y.append(by)

l_x = min(p_corners_x)
r_x = max(p_corners_x)
u_y = min(p_corners_y)
b_y = max(p_corners_y)


image=_image.copy()
for i in range(0, rows):
	for j in range(0, cols):
		image.itemset((i, j, 0), 255)
		image.itemset((i, j, 1), 255)
		image.itemset((i, j, 2), 255)

center_x = int((r_x-l_x)/2) + l_x
center_y = int((b_y-u_y)/2) + u_y

#Euclidena distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

##selection of cemtroids

# X coordinates of random centroids
C_x = np.random.randint(0, cols-10, size=n_uavs)
# Y coordinates of random centroids
C_y = np.random.randint(0, rows-10, size=n_uavs)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)

#for i in  range(0, len(C)):
#	x, y = C[i] 
#	cv2.circle(_image, (x,y), 3, (0, 255, 0), 2, 8)


## FOR GET DATA*****
cl = 52
rw = 40
size_cell_x = int(cols/cl)
size_cell_y = int(rows/rw)
cells = []
free_cells = []
cells_cost = []

def get_cost(image, x, y, size_cell_x, size_cell_y):
	average = 0
	aux_x = int(size_cell_x/2)
	aux_y = int(size_cell_y/2)
	for i in  range(y-aux_y, y+aux_y):
		for j in  range(x-aux_x, x+aux_x): 
			intensity = image[i][j]
			average = average +  intensity

	return int (average/(size_cell_x*size_cell_y))

for i in  range(0,rows, size_cell_y):
	for j in  range(0,cols, size_cell_x): 
		x = j + int(size_cell_x/2)
		y = i + int(size_cell_y/2)

		if ( (x+int(size_cell_x/2))<=cols and (y+int(size_cell_y/2))<=rows):
			cells.append((x*cols) + y)
			cells_cost.append( get_cost(gray_image, x, y, size_cell_x, size_cell_y) )

ave = reduce(lambda x, y: x + y, cells_cost) / len(cells_cost)
for i in range(0, len(cells)):
	if cells_cost[i] > ave or cells_cost[i] == ave:
		free_cells.append(cells[i])

print "free_cells:  " + str(len(free_cells))

##K-means from python
f1 = []
f2 = []
for i in range(0, len(free_cells)):
	x = int(free_cells[i]/cols)
	y = int(free_cells[i]%cols)
	f1.append(x)
	f2.append(y)

X = np.array(list(zip(f1,f2)))

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero


while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(n_uavs):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]

        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

color = []
for k in range(0, n_uavs):
	color.append((random.randint(10,240), random.randint(10,240), random.randint(10,240)))


label_cells = []
for i in range(0, n_uavs):
	points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
	#print "len points " + str(i) + " : "+str(len(points))
	for j in range(0, len(points)):
		x, y = points[j]
		c = x*cols + y
		label_cells.append((c, i))
		cv2.circle(_image, (x,y), 3, color[i], 4, 8)





#for i in  range(0, len(C)):
#	x, y = C[i]
#	cv2.circle(_image, (x,y), 3, color[i], 2, 8) 	
cv2.imwrite(filename.replace('_cropped.tif', '_kmeans.tif'),_image)

###------end K-means--------------------------------------------------------

path = []
path_end=[]
g = 0
itera = 0
#print "color -->" + str(color)
while g< len(color) : 
	label = g

	p = cells.index(label_cells[0][0])
	#path.append(cells[p])
	y_s = int(cells[p]%cols)
	row = []
	row.append(y_s)

	for i in range(0, len(label_cells)):
		p = cells.index(label_cells[i][0])
		y_f = int(cells[p]%cols)

		if y_s!= y_f:

			row.append(y_f)
		y_s = y_f

	n_row = sorted(list(set(row)))

	#print "row : " + str(n_row)
	l_poits = []
	
	for r in range(len(n_row)):
		l_poits.append([])
		for u in range(len(cells)):
			if int(cells[u]%cols)== n_row[r] and (cells[u], label) in label_cells:
				l_poits[r].append(cells[u])
				
	#print "l_poits -->  " + str(l_poits) 
	
	for r in range(len(n_row)):
		if (r%2 == 0) and len(l_poits[r])>0:
			for e in range(0, len(l_poits[r]) ):
				path.append(l_poits[r][e])
				itera+=1
		elif (r%2 != 0) and len(l_poits[r])>0:
			for e in range( len(l_poits[r])-1,-1,-1):
				path.append(l_poits[r][e])
				itera+=1

	if itera>0:
		path_end.append(path[itera-1])
	g+=1

def draw_path(path, cols, image):
	x0 = int(path[0]/cols)
	y0 = int(path[0]%cols)
	for i in range(1, len(path)):
		x1 = int(path[i]/cols)
		y1 = int(path[i]%cols)
		cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 1, 8)
		x0 = x1
		y0 = y1

	cv2.imwrite(filename.replace('_cropped.tif', '_kmeanspath.tif'), image)


uav_paths = []
#print "path-->\n" + str(path)
aux = 0
for i in range(0, n_uavs):
	uav_paths.append([])
	for j in range(aux, len(path)):
		uav_paths[i].append(path[j])
		aux = j
		if path[j] == path_end[i]:
			aux=aux+1
			break


for i in range(0, n_uavs):
	draw_path(uav_paths[i], cols, _image)


print "path_end-->\n" + str(path_end)
#print "uav_paths-->\n" + str(uav_paths)

print("--- %s seconds ---" % (time.time() - start_time))
##Create csv files for GAMA simulation
f = open(filename.replace('_cropped.tif', '_kmeans.csv'),'w')
for i in range(0, n_uavs):
	for j in range (len(uav_paths[i])):
		
		f.write(str( cells.index(uav_paths[i][j]) ) )
		f.write('\n')
f.close()

f = open(filename.replace('_cropped.tif', '_kmeans_stops.csv'),'w')
for i in range(0, len(path_end)):
		f.write(str( cells.index(path_end[i]) ) )
		f.write('\n')
f.close()
