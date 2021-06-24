import cv2 
import numpy as np
import random
from osgeo import gdal, gdalnumeric, ogr, osr

filename = "/home/liseth/MEGA/DecompositionCodes/geometry_decom/guacas_3_google_cropped.tif"
img = cv2.imread(filename)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
n_uavs = 4
height, width = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new = 255 - gray
gray = cv2.GaussianBlur(gray, (3, 3), 3)
t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
_, contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
final = np.zeros(img.shape,np.uint8)
mask = np.zeros(gray.shape,np.uint8)



i=0
color_ = []
p_critical=[]
p_corners=[]
p_aux= []
  
for c in contours:
	area = cv2.contourArea(c)
	
	if area > 500 and area<(height*width)/2:
		#cv2.drawContours(img, [c], 0, (0, 255, 0), 1, cv2.LINE_AA)
		color_.append((random.randint(10,240), random.randint(10,240), random.randint(10,240)))
		cv2.drawContours(mask, contours, i, 255, -1)
		#cv2.drawContours(final, contours, i, cv2.mean(src, mask), -1)
		cv2.drawContours(final, contours, i, color_[i], -1)
		
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		cv2.circle(final, extLeft, 4, (0, 0, 255), -1)
		cv2.circle(final, extRight, 4, (0, 255, 0), -1)
		cv2.circle(final, extTop, 4, (255, 0, 0), -1)
		cv2.circle(final, extBot, 4, (255, 255, 0), -1)
		lx, ly = extLeft
		rx, ry = extRight
		ux, uy = extTop
		bx, by = extBot

		p_corners.append([ly, lx])
		p_corners.append([ry, rx])
		p_corners.append([uy, ux])
		p_corners.append([by, bx])

		i += 1

p_corners = sorted(p_corners)

for i in p_corners: 
	p_aux.append(i[1]) 
l_x = min(p_aux)
r_x = max(p_aux)
p_aux= sorted(list(set(p_aux)))

##limits
cv2.imwrite(filename.replace('_cropped.tif', '_limits.tif'),final)


for k in p_aux:
	for i in p_aux:
		if (i>=k-int(width/10) and i<=k+int(width/10)):
			p_aux.remove(i)
			
print "p_aux" +  str(p_aux)
j = 1
p_critical.append(p_corners[0])
for k in p_aux:
	for i in range(len(p_corners)):
		if k == p_corners[i][1] and p_corners[i][1]!= p_critical[j-1][1]:
			p_critical.append(p_corners[i])
			j+=1




p_critical= sorted(p_critical)		
print "p_critical" +  str(p_critical)


image=img.copy()
for i in range(0, height):
	for j in range(0, width):
		image.itemset((i, j, 0), 255)
		image.itemset((i, j, 1), 255)
		image.itemset((i, j, 2), 255)


for r in range(0, len(p_critical)):

	if p_critical[r][1] >= l_x and p_critical[r][0]<= r_x:
		cv2.line(img, (p_critical[r][1], p_critical[r][0]), (p_critical[r][1], 0), (0, 0, 0), 1, 8)
		cv2.line(image, (p_critical[r][1], p_critical[r][0]), (p_critical[r][1], 0), (0, 0, 0), 1, 8)
		#else:
		cv2.line(img, (p_critical[r][1], p_critical[r][0]), (p_critical[r][1], height-1), (0, 0, 0), 1, 8)
		cv2.line(image, (p_critical[r][1], p_critical[r][0]), (p_critical[r][1], height-1), (0, 0, 0), 1, 8)



cv2.imwrite(filename.replace('_cropped.tif', '_trapezoid.tif'),img)
cv2.imwrite(filename.replace('_cropped.tif', '_subtrapezoid.tif'),image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
new = 255 - gray
gray = cv2.GaussianBlur(new, (3, 3), 3)
t, dst = cv2.threshold(gray, 0, 200, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)
_, contours, _ = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

cl = 60
rw = 40
size_cell_x = int(width/cl)
size_cell_y = int(height/rw)
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

for i in  range(0,height, size_cell_y):
	for j in  range(0,width, size_cell_x): 
		x = j + int(size_cell_x/2)
		y = i + int(size_cell_y/2)

		if ( (x+int(size_cell_x/2))<=width and (y+int(size_cell_y/2))<=height):
			cells.append((x*width) + y)
			cells_cost.append( get_cost(gray_image, x, y, size_cell_x, size_cell_y) )

ave = reduce(lambda x, y: x + y, cells_cost) / len(cells_cost)
for i in range(0, len(cells)):
	if cells_cost[i] >= ave:
		free_cells.append(cells[i])
			

label_cells = []

for i in range(0, len(free_cells)):     
	for n in range(0, len(color)):
		if (final.item(int(free_cells[i]%width),int(free_cells[i]/width),0) == color[n][0]) and (final.item(int(free_cells[i]%width),int(free_cells[i]/width),1) == color[n][1]) and (final.item(int(free_cells[i]%width),int(free_cells[i]/width),2) == color[n][2]):
			label_cells.append((free_cells[i], n))


#correction of cell in line of decompsition

for i in range(0, len(label_cells)-1): 
	if label_cells[i][1] == len(color)-1:
		if abs(label_cells[i+1][1]-label_cells[i-1][1])==1 :
			label_cells[i] = (label_cells[i][0], label_cells[i+1][1])
		elif int(label_cells[i+1][0]%width)== int(label_cells[i][0]%width):
			label_cells[i] = (label_cells[i][0], label_cells[i+1][1])
		elif int(label_cells[i-1][0]%width)== int(label_cells[i][0]%width):
			label_cells[i] = (label_cells[i][0], label_cells[i-1][1])
		else:
			label_cells[i] = (label_cells[i][0], label_cells[i-1][1])


print "label_cells" +str(label_cells)
		
trap = cv2.imread(filename.replace('_cropped.tif', '_trapezoid.tif'))
for i in  range(0, len(free_cells)):
	x=  int(free_cells[i]/width)
	y = int(free_cells[i]%width)
	cv2.circle(trap, (x,y), 3, color[label_cells[i][1]], 4, 8)

cv2.imwrite(filename.replace('.tif', '_labeled.tif'), final)
cv2.imwrite(filename.replace('_cropped.tif', '_trapezoidlabeled.tif'), trap)

##Doing zigzag path
print "free_cells" +str(free_cells)


path = []
path_end=[]
g = 0
itera = 0
#print "color -->" + str(color)
while g< len(color) and color != [0, 0, 0]: 
	label = g

	p = cells.index(label_cells[0][0])
	#path.append(cells[p])
	y_s = int(cells[p]%width)
	row = []
	row.append(y_s)

	for i in range(0, len(label_cells)):
		p = cells.index(label_cells[i][0])
		y_f = int(cells[p]%width)

		if y_s!= y_f:
			row.append(y_f)
		y_s = y_f

	#print "rows -->  " + str(row) 
	

	l_poits = []
	i=0
	for r in range(0, len(row)):
		l_poits.append([])
		for u in range(0, len(cells)):
			#print (cells[u], label)
			if int(cells[u]%width)== row[r] and (cells[u], label) in label_cells:
				l_poits[r].append(cells[u])
				i+=1
	print "l_poits -->  " + str(l_poits) 
	
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

	for i in range(1, len(path)):
		x1 = int(path[i]/cols)
		y1 = int(path[i]%cols)
		cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 1, 8)
		x0 = x1
		y0 = y1

	cv2.imwrite(filename.replace('_cropped.tif', '_trapepath.tif'), image)

## Small areas, delete same elements

path_end = sorted(list(set(path_end)))

uav_paths = []
print "path-->\n" + str(path)
aux = 0
for i in range(0, n_uavs):
	uav_paths.append([])
	for j in range(aux, len(path)):
		uav_paths[i].append(path[j])
		aux = j
		if path[j] == path_end[n_uavs-1-i]:
			#Toma las areas de izquierda a derecha
			aux=aux+1
			break

for i in range(0, n_uavs):
	draw_path(uav_paths[i], width, trap)




print "uav_paths-->\n" + str(uav_paths)

##Create csv file for GAMA simulation
##Create csv files for GAMA simulation
f = open(filename.replace('_cropped.tif', '_trape.csv'),'w')
for i in range(0, n_uavs):
	for j in range (len(uav_paths[i])):
		
		f.write(str( cells.index(uav_paths[i][j]) ) )
		f.write('\n')
f.close()

f = open(filename.replace('_cropped.tif', '_trape_stops.csv'),'w')
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
    


###Create mission.txt
lt = []
lg = []
for i in range(0, n_uavs):
	lt[:]=[ ]
	lg[:]=[ ]
	lt, lg = get_coordinates(width, uav_paths[i])
	f = open(filename.replace('_cropped.tif', '_'+str(i+1)+'.txt'),'w')
	f.write('QGC WPL 110')
	f.write('\n')
	f.write(str(0)+'\t'+str(1)+ '\t'+str(3)+ '\t'+str(16) + '\t'+str(0) + '\t'+str(3)+ '\t'+str(0) + '\t'+str(0) + '\t' +str(lt[0]) + '\t' +str(lg[0]) + '\t' +str(0)+ '\t' +str(1))
	f.write('\n')
	for j in range(0, len(lt)):
		f.write(str(j+1)+'\t'+str(0)+ '\t'+str(3)+ '\t'+str(16) + '\t'+str(0) + '\t'+str(3)+ '\t'+str(0) + '\t'+str(0) + '\t' +str(lt[j]) + '\t' +str(lg[j]) + '\t' +str(20)+ '\t' +str(1))
		f.write('\n')
	f.close()

    






