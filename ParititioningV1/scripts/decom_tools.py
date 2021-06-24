
#!/usr/bin/env python
import os, sys
import cv2
import numpy as np
import random
from Voronoi import Voronoi
from osgeo import gdal, gdalnumeric, ogr, osr
import threading

import math



#Start Opencv Processing

def read_imagefile(filename):
	image = cv2.imread(filename)
	rows, cols = image.shape[:2]
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #no modified

	return image, gray_image, rows, cols

#Get corners of cropped image for getting center of polygon 
def get_center(image):
	rows, cols = image.shape[:2]

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	new = 255 - gray
	gray = cv2.GaussianBlur(gray, (3, 3), 3)
	t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
	contours, h = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	final = np.zeros(image.shape,np.uint8)
	mask = np.zeros(gray.shape,np.uint8)

	i = 0

	p_corners_x=[]
	p_corners_y=[]
	
	for c in contours:
		area = cv2.contourArea(c)
		if area > 600 and area<(rows*cols):
		
			lx, ly= tuple(c[c[:, :, 0].argmin()][0])
			rx, ry = tuple(c[c[:, :, 0].argmax()][0])
			ux, uy = tuple(c[c[:, :, 1].argmin()][0])
			bx, by = tuple(c[c[:, :, 1].argmax()][0])
			
	
			p_corners_x.append(lx)
			p_corners_x.append(rx)
			p_corners_y.append(uy)
			p_corners_y.append(by)
	l_x = min(p_corners_x)
	r_x = max(p_corners_x)
	u_y = min(p_corners_y)
	b_y = max(p_corners_y)
	center_x = int((r_x-l_x)/2) + l_x
	center_y = int((b_y-u_y)/2) + u_y

	print "center: " + str(center_x) + " , " + str(center_y)

	return center_x, center_y



def get_points_circle(image, center_x, center_y, radio):
	Points = []
	rows, cols = image.shape[:2]
	for x in range(0,cols):
		for y in range(0,rows):
			if (x-center_x)**2 + (y-center_y)**2 <= radio**2 and (x-center_x)**2 + (y-center_y)**2 >= (radio//2)**2:
				Points.append((x, y))


	return Points

def select_centroids(n_uavs, center_x, center_y, Points):
	i = 0
	randPoints = []
	aux = -1
	#Select to n_uavs-1 because now is include the center
	randPoints.append((center_x, center_y))
	while i<n_uavs-1:
		n = random.randint(0, len(Points))
		if n != aux:
			x, y = Points[n]
			randPoints.append((x,y))
			i += 1
		aux = n
	#print "randPoints : " + str(randPoints)
	return randPoints

def draw_voronoi_partition(randPoints, image, filename):
	vp = Voronoi(randPoints) 
	vp.process()
	lines = vp.get_output()
	rows, cols = image.shape[:2]

	white_img = image.copy()
	
	for i in range(0, rows):
		for j in range(0, cols):
			white_img.itemset((i, j, 0), 255)
			white_img.itemset((i, j, 1), 255)
			white_img.itemset((i, j, 2), 255)

	for i in  range(0, len(lines)):
		x0, y0, x1, y1 = lines[i]
		cv2.line(white_img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), 1, 8)

	
	#cv2.imwrite(filename.replace('_cropped.tif', '_voronoi.tif'),image)
	cv2.imwrite(filename.replace('_cropped.tif', '_white_voronoi.tif'),white_img)

	return white_img
#Get contours from white image for get labeled area

def label_area(filename, white_img, free_cells, randPoints):
	rows, cols = white_img.shape[:2]
	gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
	new = 255 - gray
	gray = cv2.GaussianBlur(new, (3, 3), 3)
	t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
	contours, h = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	final = np.zeros(white_img.shape,np.uint8)
	mask = np.zeros(gray.shape,np.uint8)

	i=0
	color = [] #This color is for labeled
	labelarea = [] ##Heuristic
	for c in contours:
		area = cv2.contourArea(c)
		
		if area>0.0:
			lx, ly  = tuple(c[c[:, :, 0].argmin()][0])
			rx, ry = tuple(c[c[:, :, 0].argmax()][0])
			ux, uy = tuple(c[c[:, :, 1].argmin()][0])
			bx, by = tuple(c[c[:, :, 1].argmax()][0])
			cx = int((rx-lx)/2) + lx
			cy = int((by-uy)/2) + uy 
			color.append((random.randint(10,250), random.randint(10,254), random.randint(10,254)))
			cv2.drawContours(mask, contours, i, 255, -1)
			cv2.drawContours(final, contours, i, color[i], -1)
			labelarea.append((' ',0,area,color[i], cx,cy))
			cv2.circle(final, (cx,cy), 2, (255, 255, 255), 1, 8)
			cv2.circle(final, (cols//2, rows//2), 2, (255, 255, 255), 2, 8)
			i += 1
	color.append((0,0,0))
	
	label_cells = []
	
	#Label cells and areas wih randpoint
	for i in range(0, len(free_cells)):     
		for n in range(0, len(color)):
			r = free_cells[i]//cols
			c = free_cells[i]%cols
			if (final.item(c,r,0) == color[n][0]) and (final.item(c,r,1) == color[n][1]) and (final.item(c,r,2) == color[n][2]):
				label_cells.append((free_cells[i], n))


	#correction of cell in line of decompsition
	for i in range(0, len(label_cells)): 
		if label_cells[i][1] == len(color)-1:
			#if abs(label_cells[i+1][1]-label_cells[i-1][1])==1 and label_cells[i+1][1] != len(color):
			#	label_cells[i] = (label_cells[i][0], label_cells[i+1][1])
			if int(label_cells[i+1][0]%cols)== int(label_cells[i][0]%cols) and label_cells[i+1][1] != len(color) and  int(label_cells[i+1][0]%cols)- int(label_cells[i][0]%cols) ==1:
				label_cells[i] = (label_cells[i][0], label_cells[i+1][1])
			elif label_cells[i+1][1] == len(color) or label_cells[i-1][1] == len(color):
				label_cells[i] = (label_cells[i][0], label_cells[i-2][1])
			else:
				label_cells[i] = (label_cells[i][0], label_cells[i-1][1])

	cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea.tif'), final)

	voro = cv2.imread(filename)
	for i in  range(0, len(free_cells)):
		x=  int(free_cells[i]/cols)
		y = int(free_cells[i]%cols)
		cv2.circle(voro, (x,y), 3, color[label_cells[i][1]], 4, 8)
	cv2.imwrite(filename.replace('_cropped.tif', '_labeledvoronoi.tif'), voro)


	return color, label_cells, labelarea, voro, final


#Labeled of free cell in each voronoi partition
def get_cost(image, x, y, size_cell_x, size_cell_y):
	average = 0
	aux_x = int(size_cell_x/2)
	aux_y = int(size_cell_y/2)
	for i in  range(x-aux_x, x+aux_x):
		for j in  range(y-aux_y, y+aux_y): 
			intensity = image[i][j]
			average = average +  intensity

	return int (average/(size_cell_x*size_cell_y))
	
def get_cells(cl, rw, gray_image):
	rows, cols = gray_image.shape[:2]
	size_cell_x = int(cols/cl)
	size_cell_y = int(rows/rw)
	cells = []
	free_cells = []
	cells_cost = []
	
	for i in  range(0,rows, size_cell_y):
		for j in  range(0,cols, size_cell_x): 
			x = i + int(size_cell_x/2)
			y = j + int(size_cell_y/2)
			if ( (x+int(size_cell_x/2))<=rows and (y+int(size_cell_y/2))<=cols):
				cells.append((y*cols) + x)
				cells_cost.append( get_cost(gray_image, x, y, size_cell_x, size_cell_y) )
	
	ave = reduce(lambda x, y: x + y, cells_cost) / len(cells_cost)
	for i in range(0, len(cells)):
		if cells_cost[i] >= ave :
			free_cells.append(cells[i])

	return cells, cells_cost, free_cells


def draw_path(path, cols, image, filename):
	x0 = int(path[0]/cols)
	y0 = int(path[0]%cols)
	for i in range(1, len(path)):
		x1 = int(path[i]/cols)
		y1 = int(path[i]%cols)
		cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 1, 8)
		x0 = x1
		y0 = y1

	cv2.imwrite(filename.replace('_cropped.tif', '_vorozigzag.tif'), image)

def zigzag_path(n_uavs, color, cells, label_cells, voro, filename):
	rows, cols = voro.shape[:2]
	path = []
	path_end=[]
	g = 0
	itera = 0

	while g< len(color) and color != [0, 0, 0]: 
		label = g
		p = cells.index(label_cells[0][0])
		
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


	#Array for paths
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
		draw_path(uav_paths[i], cols, voro, filename)

	#print "path_end-->\n" + str(path_end)
	#print "uav_paths-->\n" + str(uav_paths)
	return path, path_end, uav_paths 

##Create csv files for GAMA simulation
def create_GAMAcsv(filename, n_uavs, uav_paths, path_end):
	f = open(filename.replace('_cropped.tif', '_voronoi.csv'),'w')
	for i in range(0, n_uavs):
		for j in range (len(uav_paths[i])):
			f.write(str( cells.index(uav_paths[i][j]) ) )
			f.write('\n')
	f.close()

	f = open(filename.replace('_cropped.tif', '_voro_stops.csv'),'w')
	for i in range(0, len(path_end)-1):
		f.write(str( cells.index(path_end[i]) ) )
		f.write('\n')
	f.close()



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

def create_APPmission(filename, n_uavs, uav_paths, cols):
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


def max_min_area(labelarea):
	max_area = 0
	min_area = labelarea[0][2]

	for a in range(len(labelarea)):
		if labelarea[a][2] >= max_area:
			max_area = labelarea[a][2]
		if labelarea[a][2] <= min_area:
			min_area = labelarea[a][2]
	return min_area, max_area

def distance (p0,p1):
	d = (p1[0] - p0[0])**2 +  (p1[1] - p0[1])**2
	return (math.sqrt(d))

#New POints for voronoi partition
def new_voronoicentroids(labelarea, centroids, final):
	newPoints = []
	c_point = (0,0)
	min_x = 0
	min_y = 0
	delta = 0
	color_Q = []
	color_C = (0,0,0)
	color_P = []


	for p in range(0,len(labelarea)):
		if labelarea[p][0] == 'C':
			color_C = labelarea[p][3]

		if labelarea[p][0] == 'Qmin' or labelarea[p][0] == 'Q':
			color_Q.append(labelarea[p][3])

		if labelarea[p][0] == 'P':
			color_P.append(labelarea[p][3])
	print "Color_Q:  \n" + str(color_Q) + "\n"
	for c in centroids:
		if (final.item(c[1],c[0],0) == color_C[0]) and (final.item(c[1],c[0],1) == color_C[1]) and (final.item(c[1],c[0],2) == color_C[2]):
			min_x = c[0]
			min_y = c[1]
			c_point =(min_x,min_y)
			newPoints.append((c[0],c[1]))
			print "C found: \n" + str(c)

	for c in centroids:
		for l in color_Q:
			if (final.item(c[1],c[0],0) == l[0]) and (final.item(c[1],c[0],1) == l[1]) and (final.item(c[1],c[0],2) == l[2]):
				print "Q found: \n" + str(c)
				q_point = (c[0],c[1])
				d = distance(c_point, q_point)
				delta = int(d/2)
				if c[0] - min_x >= 0 and c[1] - min_y >= 0:
					newPoints.append((c[0] - delta, c[1] - delta))
				elif c[0] - min_x >= 0 and c[1] - min_y < 0:
					newPoints.append((c[0] - delta, c[1] + delta))
				elif c[0] - min_x < 0 and c[1] - min_y < 0:
					newPoints.append((c[0] + delta, c[1] + delta))
				elif c[0] - min_x < 0 and c[1]- min_y >= 0:
					newPoints.append((c[0] + delta, c[1] - delta))
	for c in centroids:
		for l in color_P:
			if (final.item(c[1],c[0],0) == l[0]) and (final.item(c[1],c[0],1) == l[1]) and (final.item(c[1],c[0],2) == l[2]):
				newPoints.append((c[0],c[1]))

	print "Final centroids:  " + str(newPoints) + "\n"
	return newPoints


def extract_max_area(labelarea, max_area, final, filename):
	
	rows, cols = final.shape[:2]
	color_max = (0, 0 ,0)
	for a in labelarea:
		if a[2] == max_area:
			color_max = a[3]

	#extract pixeles from image arealabeled
	image_a=final.copy()
	area_image = final.copy()
	for i in range(0, rows):
		for j in range(0, cols):
			image_a.itemset((i, j, 0), 255)
			image_a.itemset((i, j, 1), 255)
			image_a.itemset((i, j, 2), 255)
	

	for i in  range(0,rows):
		for j in  range(0,cols): 
			if (area_image.item(i,j,0) == color_max[0]) and (area_image.item(i, j,1) == color_max[1]) and (final.item(i, j,2) == color_max[2]):
				cv2.circle(image_a, (j,i), 3, color_max, 4, 8)	
	cv2.imwrite(filename.replace('_cropped.tif', '_maxarea.tif'), image_a)


	return image_a, color_max

def get_centroid_max(color_max, centroids, final):

	mx = 0
	my = 0
	for c in centroids:
		if (final.item(c[1],c[0],0) == color_max[0]) and (final.item(c[1],c[0],1) == color_max[1]) and (final.item(c[1],c[0],2) == color_max[2]):
			mx = c[0]
			my = c[1]

	print "Centroid of max_area:  "+ str(mx) + " , "+ str(my)
	return mx, my


##Read lines in subvoronoi3 for detect color_max
#new_p  = cv2.imread(filename.replace('_cropped.tif', '_subvoronoi3.tif'))

def draw_newPartition(filename, new_image, white_img, image_a, color_max):
	
	rs, cs = white_img.shape[:2]

	for i in range(0, rs):
		for j in range(0, cs):
			if new_image.item(i,j,0) == 0 and  new_image.item(i,j,1) == 0 and  new_image.item(i,j,2) == 0:
				if image_a.item(i,j,0) == color_max[0] and  image_a.item(i,j,1) == color_max[1] and image_a.item(i,j,2) == color_max[2]:
					cv2.circle(white_img, (j, i), 1, (0,0,0), 1)
					
				
	cv2.imwrite(filename.replace('_cropped.tif', '_newvorowhite.tif'),white_img)
	return white_img


#Get Q and P uavs


def set_level(labelarea, c_level, partition_level):

	c_copy = []

	if partition_level == 0:
		for c in labelarea:
			c_copy.append((c[4],c[5], 0))
	if partition_level == 1:
		for l in labelarea:
			for c in range(len(c_level)):
				p_c = (c_level[c][0], c_level[c][1])
				p_l = (l[4], l[5])
				if p_c[0] == p_l[0] and p_c[1] == p_l[1] :
					c_copy.append(c_level[c])
		for l in labelarea:
			p_l = (l[4], l[5])
			if (p_l[0], p_l[1], 0) not in c_copy and (p_l[0], p_l[1], 1) not in c_copy:
				c_copy.append((p_l[0], p_l[1], 1))
	if partition_level == 2:
		for l in labelarea:
			for c in range(len(c_level)):
				p_c = (c_level[c][0], c_level[c][1])
				p_l = (l[4], l[5])
				if p_c[0] == p_l[0] and p_c[1] == p_l[1] :
					c_copy.append(c_level[c])
		for l in labelarea:
			p_l = (l[4], l[5])
			if (p_l[0], p_l[1], 0) not in c_copy and (p_l[0], p_l[1], 1) not in c_copy and (p_l[0], p_l[1], 2) not in c_copy:
				c_copy.append((p_l[0], p_l[1], 2))

	if partition_level == 3:
		for l in labelarea:
			for c in range(len(c_level)):
				p_c = (c_level[c][0], c_level[c][1])
				p_l = (l[4], l[5])
				if p_c[0] == p_l[0] and p_c[1] == p_l[1] :
					c_copy.append(c_level[c])
		for l in labelarea:
			p_l = (l[4], l[5])
			if (p_l[0], p_l[1], 0) not in c_copy and (p_l[0], p_l[1], 1) not in c_copy and (p_l[0], p_l[1], 2) not in c_copy and (p_l[0], p_l[1], 3) not in c_copy:
				c_copy.append((p_l[0], p_l[1], 3))

	if partition_level == 4:
		for l in labelarea:
			for c in range(len(c_level)):
				p_c = (c_level[c][0], c_level[c][1])
				p_l = (l[4], l[5])
				if p_c[0] == p_l[0] and p_c[1] == p_l[1] :
					c_copy.append(c_level[c])
		for l in labelarea:
			p_l = (l[4], l[5])
			if (p_l[0], p_l[1], 0) not in c_copy and (p_l[0], p_l[1], 1) not in c_copy and (p_l[0], p_l[1], 2) not in c_copy and (p_l[0], p_l[1], 3) not in c_copy and (p_l[0], p_l[1], 4) not in c_copy:
				c_copy.append((p_l[0], p_l[1], 4))

	return c_copy



def name_C(centroids, labelarea, final, rows, cols):

	captain_loc = (cols//2, rows//2)
	color_c = (0,0,0)
	aux = cols
	c = (0,0)
	for l in centroids:
		p = (l[0], l[1])
		dist = distance(captain_loc, p)
		
		if dist >0 and dist< aux:
		 	aux = dist
		 	c = (l[1], l[0])
		 	#print "near to center: " + str(c) + "\n"

	for a in range(0,len(labelarea)):
		color_p = labelarea[a][3]
		if (final.item(c[0],c[1],0) == color_p[0]) and (final.item(c[0],c[1],1) == color_p[1]) and (final.item(c[0],c[1],2) == color_p[2]):
			color_c= color_p
	
	#print "color_c:  \n" +str(color_c)
	n = 0
	
	for a in range(0,len(labelarea)):
		if labelarea[a][3] == color_c:
			n = a

	#print "near to c: " +str(n) +"--"+ str(labelarea[n][4])+ "," + str(labelarea[n][5]) +"\n"
	labelarea[n] = ('C',) + labelarea[n][1:]
	
	#print "\nC Labeled: \n" + str(labelarea) + "\n"
	return labelarea, labelarea[n][4],labelarea[n][5], c[0], c[1]



def name_Qmin(labelarea, c_x, c_y, cols, c_level, partition_level):

	aux = cols
	captain_loc = (c_x, c_y)


	for i in range(1,partition_level+1):
		aux = cols
		for c in c_level:
			p =(c[0], c[1])
			dist = distance(captain_loc, p)
			if c[2] == i and dist>0 and dist<= aux:
		 		aux =dist

		for l in range(0,len(labelarea)):
		 	p = (labelarea[l][4], labelarea[l][5])
		 	dist = distance(captain_loc, p)
		 	if dist == aux:
		 		labelarea[l] = ('Qmin',) + labelarea[l][1:]
		print "Qmin of partition " +str(partition_level)+ " :\n"+ str(labelarea) + "\n"

	return labelarea

def name_QP(labelarea, c_x, c_y, cols, c_level, partition_level):

	d = []
	captain_loc = (c_x, c_y)
	partition_level = partition_level
	
	for l in range(len(labelarea)):
		p = (labelarea[l][4], labelarea[l][5])
		dist = distance(captain_loc, p)
		if labelarea[l][0]!= 'Qmin' and dist >0:
			d.append(dist)
			

	d = sorted(d)
	#print "distance array: " + str(d)
	if partition_level == 0:
		for l in range(len(labelarea)):
			if labelarea[l][0] != 'C' :
				labelarea[l] = ('Q',) + labelarea[l][1:]

	if partition_level == 1:
		for l in range(len(labelarea)):
			p = (labelarea[l][4], labelarea[l][5])
			dist = distance(captain_loc, p)
			
			if labelarea[l][0] != 'C' and labelarea[l][0] != 'Qmin':
				if dist == d[0] or dist == d[1] or dist == d[2]:
					labelarea[l] = ('Q',) + labelarea[l][1:]
				else:
					labelarea[l] = ('P',) + labelarea[l][1:]
	if partition_level == 2:
		for l in range(len(labelarea)):
			p = (labelarea[l][4], labelarea[l][5])
			dist = distance(captain_loc, p)
			if labelarea[l][0] != 'C' and labelarea[l][0] != 'Qmin':
				if dist == d[0] or dist == d[1] or dist == d[2]:
					labelarea[l] = ('Q',) + labelarea[l][1:]
				else:
					labelarea[l] = ('P',) + labelarea[l][1:]
	if partition_level == 3:
		for l in range(len(labelarea)):
			p = (labelarea[l][4], labelarea[l][5])
			dist = distance(captain_loc, p)
			if labelarea[l][0] != 'C' and labelarea[l][0] != 'Qmin':
				if dist == d[0] or dist == d[1] or dist == d[2]:
					labelarea[l] = ('Q',) + labelarea[l][1:]
				else:
					labelarea[l] = ('P',) + labelarea[l][1:]
	if partition_level == 4:
		for l in range(len(labelarea)):
			p = (labelarea[l][4], labelarea[l][5])
			dist = distance(captain_loc, p)
			if labelarea[l][0] != 'C' and labelarea[l][0] != 'Qmin':
				if dist == d[0] or dist == d[1] or dist == d[2]:
					labelarea[l] = ('Q',) + labelarea[l][1:]
				else:
					labelarea[l] = ('P',) + labelarea[l][1:]
	print "\nFinal Labeled: \n" + str(labelarea) + "\n"
	return labelarea

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
	center_x, center_y = get_center(image)
	#Get points[] in circle center in center_x, center_y 
	points = get_points_circle(image, center_x, center_y, radio)
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
		
		labelarea, C_x, C_y, cx, cy = name_C(centroids,labelarea, final, rows, cols)
		labelarea= name_QP(labelarea, cx, cy, cols, c_level, partition_level)
		
		
		final2 = final.copy()
		for c in centroids:
			cv2.circle(final2, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea0.tif'), final2)

		new_centroids = new_voronoicentroids(labelarea, centroids, final)
		white_img = draw_voronoi_partition(new_centroids, image, filename)
		color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
		min_area, max_area = max_min_area(labelarea)
		
		c_level = set_level(labelarea, c_level,0)
		print "c_level:  " +str(c_level) 

		labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
		labelarea= name_QP(labelarea, cx, cy, cols, c_level, partition_level)

		for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea1.tif'), final)
		#Get zizag path
		#path, path_end, uav_paths = zigzag_path(n_uavs, color, cells, label_cells, voro, filename)

	return final, white_img, max_area, labelarea, new_centroids, c_level


def partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, i):
	rows, cols = final.shape[:2]
	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, final)
	center_x, center_y, = get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, radio)
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
		labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
		labelarea = name_Qmin(labelarea, cx,cy, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, cx, cy, cols, c_level, partition_level)
		new_centroids = new_voronoicentroids(labelarea, new_centroids, final)
		
		for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea0.tif'), final)
		
		white_img = draw_voronoi_partition(new_centroids, white_img, filename)
		color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
		min_area, max_area = max_min_area(labelarea)

		c_level = set_level(labelarea, c_level, i)
		print "c_level:  " +str(c_level) +"\n"

		labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
		labelarea = name_Qmin(labelarea, cx, cy, cols, c_level, partition_level)
		labelarea= name_QP(labelarea, cx, cy, cols, c_level, partition_level)

		for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
		cv2.imwrite(filename.replace('_cropped.tif', '_labeledarea1.tif'), final)

	return final, white_img, max_area, labelarea, new_centroids, c_level

def decomposition (n_uavs, image, filename, cl, rw, decom_level):
	# num of new partitions
	N = 0 
	partition_level = set_partition_level(n_uavs)
	rows, cols = image.shape[:2]
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#Get arrays
	cells, cells_cost, free_cells = get_cells(cl, rw, gray_image)
	#For each area center and partition_level
	c_level = []
	#For get points around a semicircle
	radio = int(rows/10)

	if partition_level >= 0:
		if n_uavs>4: N = 4
		else: N =n_uavs
		final, white_img, max_area, labelarea, new_centroids, c_level = partition_base(filename, image, N, n_uavs, free_cells, c_level, radio, partition_level)
		

	if partition_level >= 1:
		if n_uavs>7: N = 4
		else: N =n_uavs -3

		final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 1)
		

	if partition_level >= 2 :
		if n_uavs>10: N = 4
		else: N =n_uavs - 6
		final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 2)


	if partition_level >= 3:
		if n_uavs>13: N = 4
		else: N =n_uavs - 9
		final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 3)

	if partition_level == 4:
		N =n_uavs - 12
		final, white_img, max_area, labelarea, new_centroids, c_level = partition(filename, final, white_img, labelarea, N, n_uavs, free_cells, c_level, new_centroids, max_area, radio, partition_level, 4)


def extract_partitioned_area(maxarea_img, color_max, final):
	rs, cs = maxarea_img.shape[:2]

	for i in range(0, rs):
		for j in range(0, cs):
			if maxarea_img.item(i,j,0) != color_max[0] and  maxarea_img.item(i,j,1) != color_max[1] and  maxarea_img.item(i,j,2) != color_max[2]:
				cv2.circle(final, (j, i), 2, (0,0,0), 1)
	return final

def label_new_captain(min_area, labelarea):

	for a in range(len(labelarea)):
		if labelarea[a][2] == min_area or abs(labelarea[a][2] - min_area) <= 150:
			labelarea[a] = labelarea[a][:1]+ (1,) + labelarea[a][2:]
	return labelarea

def max_area_captain(labelarea, captain):
	max_area = 0
	for a in labelarea:
		if a[1] == 0:
			if a[2]>=max_area:
				max_area = a[2]
	return max_area

###If n_uavs> 16

def new_captains(filename, image, radio, free_cells, N, cl, rw):

	rows, cols = image.shape[:2]
	c_level = []
	center_x, center_y= get_center(image)
	points = get_points_circle(image, center_x, center_y, radio)
	centroids = select_centroids(N, center_x, center_y, points)
	white_img = draw_voronoi_partition(centroids, image, filename)
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, centroids)
	new_area, max_area = max_min_area(labelarea)
	
	
	#new_centroids =centroids
	c_level = set_level(labelarea, c_level,0)
	print "c_level:  \n" +str(c_level) +"\n"

	labelarea, C_x, C_y, cx, cy = name_C(centroids,labelarea, final, rows, cols)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level, 0)
	new_centroids = new_voronoicentroids(labelarea, centroids, final)
	white_img = draw_voronoi_partition(new_centroids, white_img, filename)
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	new_area, max_area = max_min_area(labelarea)
	
	c_level = set_level(labelarea, c_level, 0)
	print "c_level:  " +str(c_level) +"\n"

	labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level,0)


	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, maxarea_img)
	center_x, center_y= get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, radio)
	centroids = select_centroids(4, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	new_centroids.remove((mx,my))
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	labelarea =  label_new_captain(new_area, labelarea)
	#print "labelarea : \n" + str(labelarea)
	
	c_level = set_level(labelarea, c_level, 0)
	print "c_level:  " +str(c_level) +"\n"
	
	labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level, 0)
	
	max_area = max_area_captain(labelarea, 0) 
	

	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, maxarea_img)
	center_x, center_y= get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, radio)
	centroids = select_centroids(4, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	new_centroids.remove((mx,my))
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	labelarea =  label_new_captain(new_area, labelarea)
	#print "labelarea : \n" + str(labelarea)
	c_level = set_level(labelarea, c_level, 1)
	print "c_level:  " +str(c_level) +"\n"
	
	labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
	labelarea = name_Qmin(labelarea, cx,cy, cols, c_level, 1)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level, 1)
	
	
	max_area = max_area_captain(labelarea, 0) 

	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, maxarea_img)
	center_x, center_y= get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, radio)
	centroids = select_centroids(4, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	new_centroids.remove((mx,my))
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	labelarea =  label_new_captain(new_area, labelarea)
	#print "labelarea : \n" + str(labelarea)
	c_level = set_level(labelarea, c_level, 2)
	print "c_level:  " +str(c_level) +"\n"
	
	labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
	labelarea = name_Qmin(labelarea, cx,cy, cols, c_level, 2)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level, 2)

	max_area = max_area_captain(labelarea, 0) 

	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, maxarea_img)
	center_x, center_y= get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, radio)
	centroids = select_centroids(4, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	new_centroids.remove((mx,my))
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	labelarea =  label_new_captain(new_area, labelarea)
	#print "labelarea : \n" + str(labelarea)
	c_level = set_level(labelarea, c_level, 3)
	print "c_level:  " +str(c_level) +"\n"
	
	labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
	labelarea = name_Qmin(labelarea, cx,cy, cols, c_level, 3)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level, 3)

	max_area = max_area_captain(labelarea, 0) 

	maxarea_img, color_max = extract_max_area(labelarea, max_area, final, filename)
	mx, my = get_centroid_max(color_max, new_centroids, maxarea_img)
	center_x, center_y= get_center(maxarea_img)
	points = get_points_circle( maxarea_img, center_x, center_y, radio)
	centroids = select_centroids(4, center_x, center_y, points)
	new_white_img = draw_voronoi_partition(centroids, maxarea_img, filename)
	white_img = draw_newPartition(filename, new_white_img, white_img, maxarea_img, color_max)
	new_centroids = centroids + new_centroids
	new_centroids.remove((mx,my))
	color, label_cells, labelarea, voro, final = label_area(filename, white_img, free_cells, new_centroids)
	min_area, max_area = max_min_area(labelarea)
	labelarea =  label_new_captain(new_area, labelarea)
	#print "labelarea : \n" + str(labelarea)
	c_level = set_level(labelarea, c_level, 4)
	print "c_level:  " +str(c_level) +"\n"
	
	labelarea, C_x, C_y, cx, cy = name_C(new_centroids,labelarea, final, rows, cols)
	labelarea = name_Qmin(labelarea, cx,cy, cols, c_level, 4)
	labelarea= name_QP(labelarea, cx, cy, cols, c_level, 4)


	"""final = extract_partitioned_area(maxarea_img, color_max, final)
	for c in new_centroids:
			cv2.circle(final, (c[0], c[1]), 1, (255, 255, 255), 2, 8)
	cv2.imwrite(filename.replace('_cropped.tif', '_pass.tif'), final)"""
