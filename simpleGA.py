import argparse
import datetime
import json
import os
import random
import sys
import warnings
import time

from matplotlib import pyplot as plt
import scipy

from geneticAlgorithm import GeneticAlgorithm

if __name__ == '__main__':
    warnings.filterwarnings(action="ignore", category=scipy.ComplexWarning)
    # Set work directory
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    os.chdir(path)
    # Argument parsing
    parser = argparse.ArgumentParser(description='Basic Genetic Algorithm Sample')
    parser.add_argument('-cprob', '--crossoverProbability', dest='crossover_prob', action='store', 
                        default=1.0, type=float, help='Set the probability of crossover operator.')
    parser.add_argument('-mprob', '--mutationProbability', dest='mutation_prob', action='store',
                        default=0.02, type=float, help='Set the probability of mutation operator.')
    parser.add_argument('-isize', '--individualSize', dest='gene_size', action='store',
                        default=100, type=int, help='Set the size of gene.')
    parser.add_argument('-psize', '--populationSize', dest='population_size', action='store',
                        default=200, type=int, help='Set the size of population.')
    parser.add_argument('-gen', '--generation', dest='generation', action='store',
                        default=400, type=int, help='Set the maximum generation.')
    parser.add_argument('-out', '--output', dest='output', action='store',
                        default='result.json', type=str, help='Set the output filename(json).')
    args = parser.parse_args()
    # Check argument by standard output
    print(args)

    # Set arguments to variables
    crossover_prob = args.crossover_prob
    mutation_prob = args.mutation_prob
    gene_size = args.gene_size
    popSize = args.population_size
    generation = args.generation

    # Generate each GA instance
    dsmga = GeneticAlgorithm(gene_size, popSize, crossover_prob, mutation_prob)

    best_fitness = []
    mean_fitness = []
    best_chromosomes = []

    print('Dependency Structure Matrix Genetic Algorithm...', end=' ')
    dsmga.initialization()
    bf, mf, bc = dsmga.calculateFitness()
    best_fitness.append(bf)
    mean_fitness.append(mf)
    best_chromosomes.append(bc)

    # Training
    print("Start Iteration")
    bb_info = []
    for i in range(generation):
        print("Iter["+str(i)+"]: ", end="")
        # dsmga.PairWiseTournamentSelection()
        dsm = dsmga.dsmConstruction(bb_info)
        # plt.imshow(dsm, cmap='gray', vmin=0, vmax=1)
        # plt.savefig('dsm_'+str(i)+'.png')
        bb, bb_info = dsmga.dsmClustering(dsm, bb_info)
        dsmga.bbWiseCrossover(bb)
        dsmga.pointMutation()

        dsmga.combination()
        bf, mf, bc = dsmga.calculateFitness()
        # dsmga.sortingSelection()
        dsmga.PairWiseTournamentSelection()

        best_fitness.append(bf)
        mean_fitness.append(mf)
        best_chromosomes.append(bc)
        print("Best=", bf, "Mean=", mf)

    # Print total best results.
    print('Best profit:', max(best_fitness))
    print('Mean of profit:', max(mean_fitness))
    with open('dsmga_result.txt', 'a') as f:
        f.write('Best: '+str(max(best_fitness)))
        idx = best_fitness.index(max(best_fitness))
        f.write('Chromosome: '+str(best_chromosomes[idx]))

    x_type = 'dsmga'

    # Save result to json format
    result_file = args.output
    if os.path.isfile(result_file):
        with open(result_file, 'r') as f:
            jsonDict = json.load(f)
        for i in range(generation+1):
            if x_type in jsonDict[i]:
                jsonDict[i][x_type].append(best_fitness[i])
            else:
                jsonDict[i][x_type] = [best_fitness[i]]
        jsonDict[-1]['best'].append(max(best_fitness))
        jsonDict[-1]['mean'].append(max(mean_fitness))
        jsonString = json.dumps(jsonDict, indent=4)
        with open(result_file, 'w') as f:
            f.write(jsonString)
    else:
        tmp_list = []
        for i in range(generation+1):
            tmp_dict = {
                'generation' : i,
                x_type : [best_fitness[i]]
            }
            tmp_list.append(tmp_dict)
        tmp_best = {
            'best' : [min(best_fitness)],
            'mean' : [min(mean_fitness)]
        }
        tmp_list.append(tmp_best)
        jsonString = json.dumps(tmp_list, indent=4)
        with open(result_file, 'w') as f:
            f.write(jsonString)

    # Draw plot of results.
    plt.plot(range(generation+1), best_fitness, color='green')
    plt.plot(range(generation+1), mean_fitness, color='green', linestyle='--')
    plt.xlabel('generation')
    plt.ylabel('total profit')
    plt.legend([x_type+' best', x_type+' mean'])
    now = datetime.datetime.now()
    plt.savefig('result/' + now.strftime('%Y-%m-%d_%H:%M:%S') + '.png')
