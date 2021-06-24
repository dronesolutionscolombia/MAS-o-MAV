#!/usr/bin/env python
# -*- coding: utf-8 -*-

from m_objectivefunction import TSPObjectiveFunction, QAPObjectiveFunction
from m_ga import GaSolution, GeneticOperators
from m_solution import ParetoSet, ParetoFront

import numpy as np
import matplotlib.pyplot as plt
import sys

import random, math

class NSGA:
    def __init__(self, num_objectives, genetic_operators, p, q, 
                 cr=1.0, mr=0.1):
        """
        @param num_objectives: numero de objetivos
        @param genetic_operators: objeto que representa a los operadores 
                                  genéticos
        @param p: número de variables.
        @param q: número de soluciones óptimas diferentes deseadas.
        @param cr: crossover rate
        @param mr: mutation rate
        """
        self.num_objectives = num_objectives
        self.genetic_operators = genetic_operators
        self.crossover_rate = cr
        self.mutation_rate = mr
        
        #calculamos sigma_share de acuerdo a Deb 1999
        self.sigma_share = 0.5 / math.pow(float(q), 1.0/float(p))
    
    def run(self, P, num_generations):
        """
        Ejecuta el algoritmo NSGA
        
        @param P: la poblacion inicial
        @param num_generations: el numero maximo de generaciones
        """
        for i in xrange(num_generations):
            fronts = self.classify_population(P)
            self.fitness_sharing(fronts)
            del P[:]
            #juntamos los frentes para formar P
            for front in fronts.values():
                P.extend(front)
            mating_pool = self.selection(P)
            P = self.next_generation(mating_pool, len(P))

    def classify_population(self, population):
        """
        Clasifica la población en regiones no dominadas.
        En una primera pasada se calcula el primer frente. Luego, para calcular
        los demas frentes se basa en la idea de que los elementos en el nuevo
        frente solo son dominados por elementos que se encuentran en el frente
        anterior.
        
        @param population: la población a clasificar.
        """
        fronts = {}
        n = {} # {p => k} k individuos dominan a p
        S = {} # {p => [s1, ... , sn]} p domina a s1 ... sn
        for p in population:
            S[p] = []
            n[p] = 0
        
        fronts[1] = [] #primer frente
        pop_size = len(population)
        for p in population:
            for q in population:
                if p == q:
                    continue
                elif p.dominates(q):
                    S[p].append(q)
                elif q.dominates(p):
                    n[p] += 1
            if n[p] == 0:
                p.fitness = float(pop_size)
                fronts[1].append(p)
        
        #calcular los demas frentes
        i = 1
        while(len(fronts[i]) != 0):
            next_front = []
            for r in fronts[i]:
                for s in S[r]:
                    n[s] -= 1
                    if n[s] == 0:
                        next_front.append(s)
            i += 1
            fronts[i] = next_front
        return fronts
    
    def fitness_sharing(self, fronts):
        """
        Realiza el fitness sharing hasta que cada individuo tenga el 
        valor de fitness asignado.
        
        @param fronts: Diccionario de frentes
        """
        for i, front in fronts.items():
            min_dummy_fitness = 0.0
            if i > 1: # para las siguientes poblaciones se asigna el dummy 
                      # fitness como un valor un poco 
                      # menor al valor mínimo del frente anterior
                min_dummy_fitness = min([s.fitness for s in fronts[i-1]])
                min_dummy_fitness = min_dummy_fitness * 0.8
            for sol in front:
                if i > 1:
                    sol.fitness = min_dummy_fitness
                m = self.niche_count(sol, front)
                if m > 0:
                    sol.fitness = sol.fitness / m
        
    def niche_count(self, sol, front):
        """
        Calcula el niche count.
        
        @param sol: solucion del cual se desea calcular su niche count
        @param front: el frente al cual pertenece la solucion
        """
        m = 0.0
        import sys
        uppers = [0 for _ in xrange(len(sol.objectives))]
        lowers = [sys.maxint for _ in xrange(len(sol.objectives))]
        for solution in front:
            for i, v in enumerate(solution.evaluation):
                if v > uppers[i]:
                    uppers[i] = v
                if v < lowers[i]:
                    lowers[i] = v
        for r in front:
            if r == sol: continue
            sh = 0.0 #sharing function value
            dist = sol.distance(r, uppers, lowers)
            if dist <= self.sigma_share:
                sh = 1.0 - dist / self.sigma_share
            m += sh
        return m + 1
    
    def selection(self, population):
        """
        Realiza la selección y retorna el mating_pool
        """
        pool = []
        pool_size = len(population)
        probs = self.probabilities(population)
        limits = [sum(probs[:i+1]) for i in xrange(len(probs))]
        while len(pool) < pool_size:
            aux = random.random()
            for i in xrange(len(limits)):
                if aux <= limits[i]:
                    pool.append(population[i])
                    break
        return pool
    
    def probabilities(self, population):
        """
        Utiliza el fitness de cada solucion para retornar una lista de 
        probabilidades de seleccionar dicho elemento
        """
        probs = []
        total_fitness = 0.0
        for p in population:
            total_fitness += p.fitness
        for p in population:
            probs.append(p.fitness / total_fitness)
        return probs
    
    def next_generation(self, mating_pool, pop_size):
        """
        Crea la siguiente generacion a partir del mating_pool y los operadores 
        genéticos
        
        @param mating_pool: mating pool utilizada para construir la siguiente 
                            generación de individuos
        """
        Q = []
        
        #cruzamiento
        while len(Q) < pop_size:
            parents = []
            parents.append(random.choice(mating_pool))
            other = random.choice(mating_pool)
            parents.append(other)
            if random.random() < self.crossover_rate:
                children = self.genetic_operators.crossover(parents[0], parents[1])
                Q.extend(children)
            else:
                Q.extend(parents)
        
        for ind in Q:
            if random.random() < self.mutation_rate:
                self.genetic_operators.mutation(ind)
                ind.evaluation = ind.evaluate()
        return Q


    

def distance_cities(i, j, c, f):

    x1 = f[i]%c
    y1 = int(f[i]/c)
    x2 = f[j]%c
    y2 = int(f[j]/c)

    return round(math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)), 4)
    
def horizontality_cities(i, j, c, f, sx): 

    a = int(f[i]/c)
    b = int(f[j]/c)
    if abs(a-b)<=sx and (f[i]%c == f[j]%c):
        return (1)
    else:
        return(10)

def verticality_cities(i, j, c, f, sy):
    
    a = int(f[i]%c)
    b = int(f[j]%c)
    if abs(a-b)<=sy and ( int(f[i]/c) == int(f[j]/c) ):
        return (1)
    else:
        return(10)

def rounding_cities(i, j, c, f, sx, sy):

    a0 = int(f[i]/c)
    a1 = int(f[i]%c)
    b0 = int(f[j]/c)
    b1 = int(f[j]%c)

    if abs(a0-b0)==sx and abs(a1-b1) == sy:
        return (1)
    else:
        return(10)
def parse_tsp_2(num_cities, num_objectives, c, free_cells, sx, sy):
    
    mat_objs = [] # [ [ [obj 1], [ obj 2] ],[ [obj 1] , [obj 2] ] ]

    for i in xrange(num_objectives):
        mat_objs.append([])
        for j in xrange(num_cities):
            mat_objs[i].append([])
            for k in xrange(num_cities):
                if i == 0:
                    mat_objs[i][j].append(distance_cities(j, k, c, free_cells))
                if i == 1:
                    mat_objs[i][j].append(rounding_cities(j, k, c, free_cells, sx, sy))
                if i == 2:
                    mat_objs[i][j].append(horizontality_cities(j, k, c, free_cells, sx))
    #print "mat_objs: \n" + str(mat_objs[1])
    return mat_objs
# run tsp
def run_nsga(n, num_cities, c, free_cells, sx, sy, total_ind, total_generations, p, q, op, num_objectives):
   
    
    cost_mats = parse_tsp_2(num_cities, num_objectives, c, free_cells, sx, sy)
    
    objs = []
    for cost_mat in cost_mats:
        objs.append(TSPObjectiveFunction(cost_mat))

    
   
    nsga = NSGA(len(objs), op, p, q, mr=0.2)
    pareto_set = ParetoSet(None)
    
    for i in xrange(n):
        pop = []
        for i in xrange(total_ind):
            sol = range(num_cities)
            random.shuffle(sol)
            pop.append(GaSolution(sol, objs))        
        nsga.run(pop, total_generations)
        pareto_set.update(pop)
    pareto_front = ParetoFront(pareto_set)
    #pareto_front.draw()
    return pareto_set


