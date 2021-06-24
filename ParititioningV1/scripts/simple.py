#!/usr/bin/env python

# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
import random
from osgeo import gdal, gdalnumeric, ogr, osr
import time

import threading
###-----------Vertical Decomposition---------######

filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_cropped.tif"

start_time =  time.time()
_image = cv2.imread(filename)
gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
rows, cols = _image.shape[:2]
gray = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
new = 255 - gray
gray = cv2.GaussianBlur(gray, (3, 3), 3)
t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
o, contours, h = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final = np.zeros(_image.shape,np.uint8)
mask = np.zeros(gray.shape,np.uint8)
n_uavs = 16


i=0
color_ = []
p_corners=[]
r_x=0
l_x=0
lx= 0
ly=0

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
		#print(lx,rx)
		p_corners.append(uy)
		p_corners.append(by)


u_y = min(p_corners)
b_y = max(p_corners)
print(l_x,r_x)



image=_image.copy()
for i in range(0, rows):
	for j in range(0, cols):
		image.itemset((i, j, 0), 255)
		image.itemset((i, j, 1), 255)
		image.itemset((i, j, 2), 255)


label_cells = []
#y_start = 0
#y_end = rows
#step_size = int((r_x-l_x) / n_uavs)

#for x in range(l_x+step_size, r_x-step_size, step_size):
#    cv2.line(image, ((x, y_start)), ((x, y_end)), (0, 0, 0), 1, 8)
#    cv2.line(_image, ((x, y_start)), ((x, y_end)), (0, 0, 0), 1, 8)
    
x_start = 0
x_end = cols
j = 0
step_size = int((b_y-u_y) / n_uavs)
for y in range(u_y+step_size, b_y, step_size):
	#print(y, b_y, step_size)
	cv2.line(image, ((x_start, y)), ((x_end, y)), (0, 0, 0), 1, 8)
	cv2.line(_image, ((x_start, y)), ((x_end, y)), (0, 0, 0), 1, 8)
	j += 1
	if j == n_uavs -1:
		break

cv2.imwrite(filename.replace('_cropped.tif', '_simple.tif'), image)
cv2.imwrite(filename.replace('_cropped.tif', '_subsimple.tif'), _image)

gray_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
new = 255 - gray_
gray = cv2.GaussianBlur(new, (3, 3), 3)
t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
o, contours, h = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final = np.zeros(image.shape,np.uint8)
mask = np.zeros(gray.shape,np.uint8)

i=0
color = []
  
for c in contours:
	area = cv2.contourArea(c)
	cv2.drawContours(image, [c], 0, (0, 255, 0), 1, cv2.LINE_AA)
	color.append((random.randint(10,240), random.randint(10,240), random.randint(10,240)))
	cv2.drawContours(mask, contours, i, 255, -1)
	#cv2.drawContours(final, contours, i, cv2.mean(src, mask), -1)
	cv2.drawContours(final, contours, i, color[i], -1)
	i += 1
color.append((0,0,0))

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
	if cells_cost[i] >= ave + 5:
		free_cells.append(cells[i])

			
print "size" + str(int(cols/size_cell_x)) + " , " + str(int(rows/size_cell_y))
label_cells = []

for i in range(0, len(free_cells)):     
	for n in range(0, len(color)):
		if (final.item(int(free_cells[i]%cols),int(free_cells[i]/cols),0) == color[n][0]) and (final.item(int(free_cells[i]%cols),int(free_cells[i]/cols),1) == color[n][1]) and (final.item(int(free_cells[i]%cols),int(free_cells[i]/cols),2) == color[n][2]):
			label_cells.append((free_cells[i], n))

#correction of cell in line of decompsition

for i in range(0, len(label_cells)): 
	if label_cells[i][1] == len(color)-1:
		if int(label_cells[i+1][0]%cols)== int(label_cells[i][0]%cols) and label_cells[i+1][1] != len(color) and  int(label_cells[i+1][0]%cols)- int(label_cells[i][0]%cols) ==1:
			label_cells[i] = (label_cells[i][0], label_cells[i+1][1])
		elif label_cells[i+1][1] == len(color) or label_cells[i-1][1] == len(color):
			label_cells[i] = (label_cells[i][0], label_cells[i-2][1])
		else:
			label_cells[i] = (label_cells[i][0], label_cells[i-1][1])
			


simple = cv2.imread(filename.replace('_cropped.tif', '_subsimple.tif'))
for i in  range(0, len(free_cells)):
	x=  int(free_cells[i]/cols)
	y = int(free_cells[i]%cols)
	cv2.circle(simple, (x,y), 3, color[label_cells[i][1]], 4, 8)

#cv2.imwrite(filename.replace('.tif', '_simplelabeled.tif'), final)
cv2.imwrite(filename.replace('_cropped.tif', '_simplelabeled.tif'), simple)




path = []
path_end=[]
g = 0
itera = 0
#print "color -->" + str(color)
while g< len(color) and color != [0, 0, 0]: 
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

	#print "rows -->  " + str(row) 
	

	l_poits = []
	i=0
	for r in range(len(row)):
		l_poits.append([])
		for u in range(len(cells)):
			if int(cells[u]%cols)== row[r] and (cells[u], label) in label_cells:
				l_poits[r].append(cells[u])
				i+=1
	#print "l_poits -->  " + str(l_poits) 
	
	for r in range(len(row)):
		if (r%2 == 0) and len(l_poits[r])>0:
			for e in range(len(l_poits[r])):
				path.append(l_poits[r][e])
				itera+=1
		elif (r%2 != 0) and len(l_poits[r])>0:
			for e in range(len(l_poits[r])-1, -1, -1):
				path.append(l_poits[r][e])
				itera+=1

	if itera>0:
		path_end.append(path[itera-1])
	g+=1

def draw_path(path, cols, image):
	x0 = int(path[0]/cols)
	y0 = int(path[0]%cols)
	#print "cols--"+str(cols)
	for i in range(1, len(path)):
		x1 = int(path[i]/cols)
		y1 = int(path[i]%cols)
		cv2.line(simple, (x0, y0), (x1, y1), (255, 0, 0), 1, 8)
		x0 = x1
		y0 = y1

	cv2.imwrite(filename.replace('_cropped.tif', '_simplepath.tif'), simple)

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
#print "uav_paths" + str(uav_paths)


print("--- %s seconds ---" % (time.time() - start_time))
##Create csv files for GAMA simulation
f = open(filename.replace('_cropped.tif', '_simple.csv'),'w')
for i in range(0, n_uavs):
	for j in range (len(uav_paths[i])):
		
		f.write(str( cells.index(uav_paths[i][j]) ) )
		f.write('\n')
f.close()

f = open(filename.replace('_cropped.tif', '_simple_stops.csv'),'w')
for i in range(0, len(path_end)-1):
		f.write(str( cells.index(path_end[i]) ) )
		f.write('\n')
f.close()

for i in range(0, n_uavs):
	draw_path(uav_paths[i], cols, simple)

##Get lat and long from image
def get_coordinates(columns, path):
	lat = []
	lon = []
	srcimage = gdal.Open(filename.replace('_cropped.tif', '.tif'))
	xoff, a, b, yoff, d, e  = srcimage.GetGeoTransform()

	for i in  range(0, len(path)):
		x = int(path[i] / columns)
		y = int(path[i] % columns)
		xp,yp = pixel2coord(x,y, xoff, a, b, yoff, d, e)
		#print "coordinate:   " + str(yp)+","+str(xp)
		lat.append(yp)
		lon.append(xp)

	return lat, lon
    
def pixel2coord(x, y, xoff, a, b, yoff, d, e):
	xp = (a * x) + (b * y) + xoff
	yp = (d * x) + (e * y) + yoff

	return(xp, yp)

###Create mission.txt
lt = []
lg = []
for i in range(0, n_uavs):
	lt[:]=[ ]
	lg[:]=[ ]
	lt, lg = get_coordinates(cols, uav_paths[i])
	f = open(filename.replace('_cropped.tif', '_'+str(i+1)+'.txt'),'w')
	f.write('QGC WPL 110')
	f.write('\n')
	f.write(str(0)+'\t'+str(1)+ '\t'+str(3)+ '\t'+str(16) + '\t'+str(0) + '\t'+str(3)+ '\t'+str(0) + '\t'+str(0) + '\t' +str(lt[0]) + '\t' +str(lg[0]) + '\t' +str(0)+ '\t' +str(1))
	f.write('\n')
	for j in range(0, len(lt)):
		f.write(str(j+1)+'\t'+str(0)+ '\t'+str(3)+ '\t'+str(16) + '\t'+str(0) + '\t'+str(3)+ '\t'+str(0) + '\t'+str(0) + '\t' +str(lt[j]) + '\t' +str(lg[j]) + '\t' +str(20)+ '\t' +str(1))
		f.write('\n')
	f.close()





