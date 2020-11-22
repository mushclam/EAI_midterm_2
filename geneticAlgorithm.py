import random
import math
import copy
import time
from matplotlib import pyplot as plt
from scipy import special as sp
import numpy as np

from individual import Individual

class GeneticAlgorithm():
    def __init__(self, geneSize, populationSize, crossoverProb, mutationProb):
        # geneSize: gene size of Individual
        # populationSize: total number of genes
        # crossoverProb: the probability of crossover operator
        # mutationProb: the probability of mutation operator
        # population: the list of individuals(geneSize, gene, totalWeight, totalProfit)
        # best: the best profit of population
        # mean: mean value of fitness of population
        self.geneSize = geneSize
        self.populationSize = populationSize
        self.crossoverProb = crossoverProb
        self.mutationProb = mutationProb
        self.population = []
        self.offspring = []
        self.best = 0
        self.mean = 0

    def initialization(self):
        # Generate random sampled genes of a population size
        # gene: bit-string of gene size
        for _ in range(self.populationSize):
            gene = Individual(self.geneSize)
            gene.initialization()
            self.population.append(gene)
        print(self.populationSize, 'genes are generated!')

    def kspRouletteWheelSelection(self):
        tmp_pop = []
        roulette = 0

        # Roulette wheel selection with raw profit value show bad selection performance,
        # because profits have large and not normalized value.
        # To decrease the unbalanced proportion of roulette,
        # normalization technique, max-min scaling is used.
        # It normalize profit values from 0 to 1.
        # For this, get max and min of fitness set.

        # Max-min scaling
        # for i, Individual in enumerate(self.population):
        #     if Individual.totalDist != 0:
        #         Individual.totalDist = fmax - Individual.totalDist
        #         self.fitness[i] = Individual.totalDist

        # fmax = max(self.fitness)
        # fmin = min(self.fitness)

        for Individual in self.population:
            # Individual.totalDist = (Individual.totalDist - fmin) / (fmax - fmin)
            roulette += Individual.totalDist

        # Roulette Wheel Selection with max-min scaling
        for i in range(self.populationSize):
            pivot = random.random()
            k = 0
            slot = 0
            while k < len(self.population) and pivot > (slot / roulette):
                slot += self.population[k].totalDist
                k += 1

            k -= 1
            tmp_pop.append(self.population[k])

        self.population = tmp_pop

    def PairWiseTournamentSelection(self):
        # Pair-wise Tournament Selection
        tmp_pop = []

        for Individual in self.population:
            competitive = self.population[random.randint(0, self.populationSize-1)]
            if Individual.fitness > competitive.fitness:
                tmp_pop.append(Individual)
            else:
                tmp_pop.append(competitive)

        self.population = tmp_pop

    def rankingSelection(self, phi):
        tmp_pop = []
        selectionProb = {}
        alpha = (2 * self.populationSize - phi * (self.populationSize + 1)) / (self.populationSize * (self.populationSize - 1))
        beta = (2 * (phi - 1)) / (self.populationSize * (self.populationSize - 1))

        sortedFitness = sorted(self.fitness.items(), key=(lambda x:x[1]), reverse=True)
        for i, item in enumerate(sortedFitness):
            selectionProb[item[0]] = alpha + beta * (i+1)

        for i in range(self.populationSize):
            pivot = random.random()
            sum_prob = 0
            idx = 0
            while pivot > sum_prob:
                sum_prob += selectionProb[sortedFitness[idx][0]]
                idx += 1

            idx -= 1    
            tmp_pop.append(self.population[sortedFitness[idx][0]])

    def sortingSelection(self):
        tmp_pop = []
        count = 0
        for fit in self.sortedFit:
            tmp_pop.append(self.population[fit[0]])
            count += 1
            if count >= self.populationSize:
                break
        self.population = tmp_pop

    def bbWiseCrossover(self, building_block):
        tmp_pop = []
        random.shuffle(self.population)

        if random.random() <= self.crossoverProb:
            # bb = []
            # while len(bb) < 3:
            #     tmp_bb = building_block[random.randint(0, len(building_block)-1)]
            #     if tmp_bb not in bb:
            #         bb.append(tmp_bb)
            bb = building_block

            overlap = []
            for i in range(len(bb) - 1):
                if len(bb[i] & bb[i+1]) <= 0:
                    overlap.append(None)
                else:
                    overlap.append(random.sample((bb[i] & bb[i+1]), 1)[0])

            orig_gene = [ind.gene for ind in self.population]
            idx = [list(b) for b in bb]

            for i in range(len(overlap)):
                next_gene = []
                if overlap[i] is not None:
                    for j in range(2):
                        p = [np.array(gene) for gene in orig_gene if gene[overlap[i]] == j]
                        cut_p = [gene[idx[i]] for gene in p]
                        random.shuffle(p)

                        for k in range(len(p)):
                            tmp_p = copy.deepcopy(p[k])
                            tmp_p[idx[i]] = cut_p[k]
                            next_gene.append(tmp_p.tolist())
                else:
                    p = [np.array(gene) for gene in orig_gene]
                    cut_p = [gene[idx[i]] for gene in p]
                    random.shuffle(p)

                    for j in range(len(p)):
                        tmp_p = copy.deepcopy(p[j])
                        tmp_p[idx[i]] = cut_p[j]
                        next_gene.append(tmp_p.tolist())
                orig_gene = next_gene

            for gene in next_gene:
                n_p = Individual(self.geneSize)
                n_p.initialization(gene)
                tmp_pop.append(n_p)
            
        self.population = tmp_pop

    def pointMutation(self):
        for Individual in self.population:
            for i in range(self.geneSize):
                if random.random() < self.mutationProb:
                    Individual.gene[i] = 1 - Individual.gene[i]

    def dsmConstruction(self, bb_info):
        if len(bb_info) != 0:
            m = len(bb_info)
            k = sum([sum(x) for x in bb_info]) / len(bb_info)
        else:
            m = 100
            k = 1

        c = 1 / (8 * math.pi * (k**2))
        l = k * m
        # _threshold = 0.03
        threshold = 1/(2*self.populationSize) + math.sqrt(sp.lambertw(c * (l**6)) / (2*(self.populationSize**2)))
        # __threshold = 1/(2 * self.populationSize) + math.sqrt(sp.lambertw(0.04 * (25**6))/(2*(self.populationSize**2)))
        dsm = []
        p_zero = []
        p_one = []
        for i in range(self.geneSize):
            zero_counter = 0
            one_counter = 0
            for individual in self.population:
                if individual.gene[i] == 0:
                    zero_counter += 1
                else:
                    one_counter += 1
            p_zero.append(zero_counter)
            p_one.append(one_counter)
        p_zero = [p/self.populationSize for p in p_zero]
        p_one = [p/self.populationSize for p in p_one]

        for i in range(self.geneSize):
            row = []
            for j in range(self.geneSize):
                if i == j:
                    row.append(1)
                    continue
                p_zz = 0
                p_zo = 0
                p_oz = 0
                p_oo = 0
                for individual in self.population:
                    if individual.gene[i] == 0 and individual.gene[j] == 0:
                        p_zz += 1
                    elif individual.gene[i] == 0 and individual.gene[j] == 1:
                        p_zo += 1
                    elif individual.gene[i] == 1 and individual.gene[j] == 0:
                        p_oz += 1
                    elif individual.gene[i] == 1 and individual.gene[j] == 1:
                        p_oo += 1
                    else:
                        raise Exception("Error: Individual gene contain wrong allele!")
                p_zz = p_zz / self.populationSize
                p_zo = p_zo / self.populationSize
                p_oz = p_oz / self.populationSize
                p_oo = p_oo / self.populationSize

                if not p_zz or not p_zo or not p_oz or not p_oo:
                    kld = - math.inf
                else:
                    kld = p_zz * math.log(p_zz/(p_zero[i]*p_zero[j]))
                    + p_zo * math.log(p_zo/(p_zero[i]*p_one[j]))
                    + p_oz * math.log(p_oz/(p_one[i]*p_zero[j]))
                    + p_oo * math.log(p_oo/(p_one[i]*p_one[j]))
                row.append(1 if kld > threshold else 0)
            dsm.append(row)
        return dsm

    def dsmClustering(self, dsm, bb_info):
        max_cluster = 50
        chromosome = []
        # Make Initinal DSM
        if bb_info:
            chromosome = bb_info
        else:
            # for _ in range(max_cluster):
            #     node = [random.randint(0, 1) for _ in range(self.geneSize)]
            #     chromosome.append(node)
            #
            # for i in range(self.geneSize):
            #     cluster = set([])
            #     for j in range(i+1, self.geneSize):
            #         if dsm[i][j] == 1 or dsm[j][i] == 1:
            #             cluster.update([i, j])
            #     flag = True
            #     for cl in chromosome:
            #         if cluster.issubset(cl):
            #             flag = False
            #             break
            #     if flag:
            #         chromosome.append(cluster)

            # t_ch = []
            # for cluster in chromosome:
            #     row = []
            #     for i in range(self.geneSize):
            #         if i in cluster:
            #             row.append(1)
            #         else:
            #             row.append(0)
            #     t_ch.append(row)
            # chromosome = t_ch
            #
            for i in range(self.geneSize):
                node = [0 for _ in range(self.geneSize)]
                node[i] = 1
                chromosome.append(node)

        # Update DSM information
        nn = self.geneSize  # number of node(gene)
        nc = 0              # number of cluster
        cl = []             # number of element in each cluster
        nodesets = []
        pairsets = []

        for cluster in chromosome:
            nodeset = set([])
            pairset = set([])
            tmp = sum(cluster)
            if tmp != 0:
                cl.append(tmp)
                nc += 1
            for i in range(len(cluster)):
                for j in range(len(cluster)):
                    if i == j:
                        if cluster[i] == 1:
                            nodeset.add(i)
                        continue
                    if cluster[i] == 1 and cluster[j] == 1:
                        pairset.add((i, j))
                        nodeset.add(i)
                        nodeset.add(j)
            pairsets.append(pairset)
            nodesets.append(nodeset)

        dsm_prime = self.dsmPrime(nn, pairsets)

        # Evaluate DSM
        b_eval = self.dsmFitness(self.geneSize, nc, cl, dsm_prime, dsm)

        # Hill climbing of DSM
        max_iter = int((sum(cl) / len(cl)) * nn*10) * 100

        count = 0
        for i in range(max_iter):
            # Create new chromosome
            n_chromosome = copy.deepcopy(chromosome)
            n_cl = copy.deepcopy(cl)
            n_nc = nc

            for i in range(1):
                # Select one gene
                r_idx = random.randint(0, nc-1)
                c_idx = random.randint(0, self.geneSize-1)
                # Flip the gene
                n_chromosome[r_idx][c_idx] = 1 - n_chromosome[r_idx][c_idx]

                # update number of element in cluster
                if n_chromosome[r_idx][c_idx] == 0:
                    changed = set([pair for pair in pairsets[r_idx] if c_idx in pair])
                    remove_pair = []
                    for pair in changed:
                        flag = True
                        for i, pairset in enumerate(pairsets):
                            if i == r_idx:
                                continue
                            if pair in pairset:
                                flag = False
                                break
                        if flag:
                            remove_pair.append(pair)
                    n_cl[r_idx] -= 1
                    n_dsm_prime = self.dsmRemove(dsm_prime, remove_pair)
                else:
                    changed = set([])
                    for node in nodesets[r_idx]:
                        changed.add((c_idx, node))
                        changed.add((node, c_idx))
                    n_cl[r_idx] += 1
                    n_dsm_prime = self.dsmUpdate(dsm_prime, changed)
                # Update number of cluster
                if sum(n_chromosome[r_idx]) == 0:
                    n_nc -= 1
                # Update dsm prime

            n_eval = self.dsmFitness(self.geneSize, n_nc, n_cl, n_dsm_prime, dsm)
            if n_eval <= b_eval:  
                if n_chromosome[r_idx][c_idx] == 0:
                    pairsets[r_idx] = pairsets[r_idx] - changed
                    nodesets[r_idx].remove(c_idx)
                    if sum(n_chromosome[r_idx]) == 0:
                        del n_chromosome[r_idx]
                        del pairsets[r_idx]
                        del nodesets[r_idx]
                else:
                    pairsets[r_idx].update(changed)
                    nodesets[r_idx].add(c_idx)
                chromosome, b_eval = n_chromosome, n_eval
                cl, nc = n_cl, n_nc
                dsm_prime = n_dsm_prime
                count = 0
            else:
                count += 1

            if count > 100:
                break

        building_block = copy.deepcopy(nodesets)
        for i in range(len(nodesets)):
            for j in range(len(nodesets)):
                if i == j:
                    continue
                if nodesets[i].issubset(nodesets[j]):
                    building_block.remove(nodesets[i])
                    break

        bb_info = []
        for i in range(len(building_block)):
            cluster = [0 for _ in range(self.geneSize)]
            for node in building_block[i]:
                cluster[node] = 1
            bb_info.append(cluster)

        plt.imshow(dsm_prime, cmap='gray', vmin=0, vmax=1)
        plt.savefig('dsm_prime.png')

        # no_overlap = []
        # for i in range(len(nodesets)):
        #     flag = True
        #     for j in range(len(nodesets)):
        #         if i == j:
        #             continue
        #         if len(nodesets[i] & nodesets[j]) != 0:
        #             flag = False
        #             break
        #     if flag:
        #         no_overlap.append(nodesets[i])

        return building_block, bb_info

    def dsmClustering2(self, dsm, bb_info):
        pairsets = []
        pairset = set([])
        for i in range(len(dsm)):
            for j in range(len(dsm)):
                if i == j:
                    continue
                if dsm[i][j] == 1:
                    pairset.add(i, j)
        pairsets.append(pairset)

        for i in range(100):
            n_dsm = copy.deepcopy(dsm)
            idx = random.randint(0, len(n_dsm)-1)

    def dsmFitness(self, nn, nc, cl, dsm_prime, dsm):
        alpha = 0.3
        beta = 0.3
        s1 = []
        s2 = []

        t_dsm = np.array(dsm)
        t_dsm_prime = np.array(dsm_prime)

        s1 = np.logical_and(np.where(t_dsm==0, 1, 0), t_dsm_prime)
        s2 = np.logical_and(t_dsm, np.where(t_dsm_prime==0, 1, 0))
        
        return (1-alpha-beta) * (nc*math.log2(nn) + math.log2(nn)*sum(cl)) + alpha*(np.sum(s1)*(2*math.log2(nn)+1)) + beta*(np.sum(s2)*(2*math.log2(nn)+1))

    def dsmPrime(self, nn, pairsets):
        dsm_prime = [[0 for _ in range(nn)] for _ in range(nn)]
        for pairset in pairsets:
            for pair in pairset:
                i, j = pair
                dsm_prime[i][j] = 1

        for i in range(nn):
            dsm_prime[i][i] = 1

        return dsm_prime

    def dsmUpdate(self, dsm, pairset):
        tmp_dsm = copy.deepcopy(dsm)
        for pair in pairset:
            i, j = pair
            tmp_dsm[i][j] = 1
            tmp_dsm[j][i] = 1
        return tmp_dsm

    def dsmRemove(self, dsm, pairset):
        tmp_dsm = copy.deepcopy(dsm)
        for pair in pairset:
            i, j = pair
            tmp_dsm[i][j] = 0
            tmp_dsm[j][i] = 0
        return tmp_dsm

    def dsmArrange(self, dsm, bb):
        tmp_dsm = copy.deepcopy(dsm)
        tmp_dsm = np.array(tmp_dsm)
        for b in bb:
            tmp = tmp_dsm[list(b)]
            tmp_dsm = np.delete(tmp_dsm, list(b), axis=0)
            tmp_dsm = np.concatenate((tmp, tmp_dsm), axis=0)
        for b in bb:
            tmp = tmp_dsm[:, list(b)]
            tmp_dsm = np.delete(tmp_dsm, list(b), axis=1)
            tmp_dsm = np.concatenate((tmp, tmp_dsm), axis=1)

        return tmp_dsm

    def combination(self):
        self.population = self.population + self.offspring
        self.offspring = []

    def getElite(self):
        self.elite = self.population[self.sortedFit[0][0]]

    def setElite(self):
        if self.elite:
            worst_key = self.sortedFit[len(self.sortedFit)-1][0]
            self.population[worst_key] = self.elite

    def calculateFitness(self):
        self.fitness = {}
        for i, gene in enumerate(self.population):
            gene.evaluation()
            self.fitness[i] = gene.fitness

        self.sortedFit = sorted(self.fitness.items(), key=(lambda x:x[1]), reverse=True)
        self.bestChromosome = self.population[self.sortedFit[0][0]]
        self.best = self.sortedFit[0][1]
        self.mean = sum(self.fitness.values()) / len(self.fitness)

        return self.best, self.mean, self.bestChromosome

    def offspringCalculateFitness(self, salesman):
        self.o_fitness = {}
        for i, gene in enumerate(self.offspring):
            gene.evaluation(salesman)
            self.o_fitness[i] = gene.totalDist

        self.o_sortedFit = sorted(self.o_fitness.items(), key=(lambda x:x[1]), reverse=True)
        self.o_best = self.o_sortedFit[0][1]
        self.o_mean = sum(self.o_fitness.values()) / len(self.o_fitness)

        return self.o_best, self.o_mean

    def printMaxSolution(self):
        print(max(self.fitness.values()))

    def printMinSolution(self):
        print(min(self.fitness.values()))
