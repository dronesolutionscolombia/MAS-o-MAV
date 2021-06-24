#!/usr/bin/env python
# -*- coding: utf-8 -*-


from m_moaco import *
from m_ant import *
import math
import random

class Moacs(Moaco):
	def __init__(self, qsubzero, tausubzero, beta, rho, cost_mats, total_ants, total_generations):
		Moaco.__init__(self, beta, rho, cost_mats, total_ants, total_generations)
		self.qsubzero = qsubzero
		self.tausubzero = tausubzero
		self.ferom_mat = []
		n = len(cost_mats[0])
		for i in xrange(n):
			self.ferom_mat.append([tausubzero for j in range(n)]) #inicializar feromonas
		self.objectives = []
		self.max_values = []


	def run(self):
		for g in xrange(self.total_generations):
			for ant_number in xrange(self.total_ants):
				ant = MOACSAnt(self.beta, ant_number, self.total_ants, self.ferom_mat, self.visib_mats, \
					self.objectives, self.tausubzero, self.qsubzero, self.rho)
				solution = ant.build_solution()
				self.pareto_set.update([solution])
				product = 1
				
				for objective_number in xrange(len(ant.average_obj)):
					product = product * ant.average_obj[objective_number]
				tausubzerop = 1 / len(self.ferom_mat) * product #len ferom_mat es la cant de nodos

				if(tausubzerop > self.tausubzero):
					self.tausubzero = tausubzerop
					reinitialize_ferom_mat()
				else:
					self.global_updating(product)

		return self.pareto_set


	def global_updating(self, product):
		for solution in self.pareto_set.solutions: #solution es una lista que tiene cada nodo de la solucion como elemento
			for i in xrange(len(solution.solution)-1):
				s = solution.solution[i]
				d = solution.solution[i+1]
				self.ferom_mat[s][d] = (1 - self.rho) * self.ferom_mat[s][d] + self.rho / product


	def reinitialize_ferom_mat(self):
		n = len(ferom_mat)
		for i in xrange(n):
			self.ferom_mat.append([self.tausebzero for j in range(n)])

class TspMoacs(Moacs):
	def __init__(self, qsubzero, tausubzero, beta, rho, cost_mats, total_ants, total_generations):
		Moacs.__init__(self, qsubzero, tausubzero, beta, rho, cost_mats, total_ants, total_generations)
		for cost_mat in cost_mats:
			self.objectives.append(TSPObjectiveFunction(cost_mat)) #construye matrices de objetivos (distancias)

class QapMoacs(Moacs):
	def __init__(self, qsubzero, tausubzero, beta, rho, cost_mats, total_ants, total_generations, dist_mat):
		Moacs.__init__(self, qsubzero, tausubzero, beta, rho, cost_mats, total_ants, total_generations)
		for cost_mat in cost_mats:
		    self.objectives.append(QAPObjectiveFunction(dist_mat, cost_mat))


def distance_cities(i, j, c, f):

    x1 = f[i]%c
    y1 = int(f[i]/c)
    x2 = f[j]%c
    y2 = int(f[j]/c)

    return round(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)), 4)
    
"""def horizontality_cities(i, j, c, f, sx): 

    a = int(f[i]/c)
    b = int(f[j]/c)
    if abs(a-b)<=sx and (f[i]%c == f[j]%c):
        return (1)
    else:
        return(10)"""

def turns(i, j, c, f):

    a1 = int(f[i]%c)
    b1= int(f[j]%c)
    a2 = int(f[i]/c)
    b2 = int(f[j]/c)

    #the same row
    if a2 == b2:
    	return(abs(a1-b1)*2)
    elif a1 == b1:
    	#return(abs(a2-b2)*1)
    	return (abs(a2-b2))
    else:
        #return( 2*math.sqrt(math.pow(a2, 2) + math.pow(b2, 2)) )
        return( abs(a2-b2)*2)

def edge(i, j, c, f):

    a1 = int(f[i]%c)
    b1= int(f[j]%c)
    a2 = int(f[i]/c)
    b2 = int(f[j]/c)

    #the same row
    if a2 == b2:
    	return(abs(a2-b2))
    elif a1 == b1:
    	return(abs(a1-b1)*2)
    	#return (0)
    else:
        #return( 2*math.sqrt(math.pow(a2, 2) + math.pow(b2, 2)) )
        return( abs(a1-b2)*2)

"""def rounding_cities(i, j, c, f, sx, sy):

    a0 = int(f[i]/c)
    a1 = int(f[i]%c)
    b0 = int(f[j]/c)
    b1 = int(f[j]%c)

    if abs(a0-b0)==sx and abs(a1-b1) == sy:
        return (1)
    else:
        return(10)"""

def parse_tsp_2(num_cities, num_objectives, c, free_cells):
    
    mat_objs = [] # [ [ [obj 1], [ obj 2] ],[ [obj 1] , [obj 2] ] ]

    for i in xrange(num_objectives):
        mat_objs.append([])
        for j in xrange(num_cities):
            mat_objs[i].append([])
            for k in xrange(num_cities):
                if i == 0:
                    mat_objs[i][j].append(distance_cities(j, k, c, free_cells))
                if i == 1:
                    mat_objs[i][j].append(turns(j, k, c, free_cells))
    #print "mat_objs: \n" + str(mat_objs[1])
    return mat_objs

# run tsp moaco
def run_moaco(n, num_cities, c, free_cells,beta, rho, qsubzero, tausubzero, total_ants, total_generations, num_objectives ):
	

	
	cost_mats = parse_tsp_2(num_cities, num_objectives, c, free_cells)
	tspMoacs = TspMoacs(qsubzero, tausubzero, beta, rho, cost_mats, total_ants, total_generations)
	

	pareto_set = ParetoSet(None)
	for i in xrange(n):
		result = tspMoacs.run()
		pareto_set.update(result.solutions)
	pareto_front = ParetoFront(pareto_set)
	#pareto_front.draw()
	return pareto_set


if __name__ == '__main__':
	testTsp()
