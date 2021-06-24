
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import spea
import random, math
import time
from m_solution import *
from m_ga import *
import cv2




filename = "/home/liseth/MEGA/MultiObjectiveCodes/guacas_3_google_cropped.tif"
image = cv2.imread(filename)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
rows, cols = image.shape[:2]
cl = 5
rw = 4
size_cell_x = int(cols/cl)
size_cell_y = int(rows/rw)
#print "cols, rows: " +str(cols)+" , "+ str(rows)
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



def get_cells (rows, cols, size_cell_x, size_cell_y, gray_image):
	cells_cost = []
	cells = []

	for i in  range(0,rows, size_cell_y):
		for j in  range(0,cols, size_cell_x): 
			x = j + int(size_cell_x/2)
			y = i + int(size_cell_y/2)
			if ( (x+int(size_cell_x/2))<=cols and (y+int(size_cell_y/2))<=rows):
				cells.append((x*cols) + y)
				cells_cost.append( get_cost(gray_image, x, y, size_cell_x, size_cell_y) )

	ave = reduce(lambda x, y: x + y, cells_cost) / len(cells_cost)
	for i in range(0, len(cells)):
		if cells_cost[i] >= ave :
			free_cells.append(cells[i])

	return cells, free_cells
		
def draw_path(path, cols, image, color, free_cells, k):
	a = free_cells[path[0]]
	x0 = int(a/cols)
	y0 = int(a%cols)
	for i in range(1, len(path)):
		b = free_cells[path[i]]
		x1 = int(b/cols)
		y1 = int(b%cols)
		cv2.line(image, (x0, y0), (x1, y1), color, 2, 8)
		x0 = x1
		y0 = y1

	cv2.imwrite(filename.replace('_cropped.tif', '_path'+str(k)+'.tif'), image)


cells, free_cells =  get_cells(rows, cols, size_cell_x, size_cell_y, gray_image)
for i in  range(0, len(free_cells)):
	x=  int(free_cells[i]/cols)
	y = int(free_cells[i]%cols)
	cv2.circle(image, (x,y), 2, (0,0,0), 4, 8)

cv2.imwrite(filename.replace('_cropped.tif', '_freecells.tif'), image)

#Parameters
n_paret_set = 5
num_cities = len(free_cells)
n_individuals = 20000
n_generations = 5000
max_pareto_size = 50
paths = []
gen_op = GeneticOperators()
num_objectives = 2

print (time.strftime("%H:%M:%S"))
pareto_set_true = ParetoSet(None)
pareto_set_spea = spea.run_spea(n_paret_set, num_cities, cols, free_cells, size_cell_x, size_cell_y, n_individuals, n_generations, max_pareto_size, gen_op, num_objectives)

pareto_front_spea = ParetoFront(pareto_set_spea)
pareto_set_true.update(pareto_set_spea.solutions)

pareto_front_true = ParetoFront(pareto_set_true)
#pareto_front_true.draw()
print (time.strftime("%H:%M:%S"))

for i in range(len(pareto_set_true.solutions)):
    obj1, obj2 = pareto_set_true.solutions[i].evaluate()
    sol = pareto_set_true.solutions[i].solution
    paths.append(sol)
    print "solution - (obj1, obj2): \n" + str(sol) + " - (" + str(obj1) + " , " +str(obj2)+ " , " +str(obj2) + ")"
print "selected solution: \n" + str(pareto_front_true.pareto_front)


for i in range(0, len(paths)):
	image_c = image.copy()
	color = (random.randint(50,240),random.randint(50,240),random.randint(50,240))
	draw_path(paths[i], cols, image_c, color, free_cells, i)