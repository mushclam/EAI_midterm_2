import argparse
import datetime
import json
import os
import random
import sys

from matplotlib import pyplot as plt

from geneticAlgorithm import GeneticAlgorithm
from salesman import Salesman

if __name__ == '__main__':
    # Set work directory
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    os.chdir(path)
    # Argument parsing
    parser = argparse.ArgumentParser(description='Basic Genetic Algorithm Sample')
    parser.add_argument('-p', '--problem', dest='problem', action='store',
                        default='TSP', type=str, help='Choose the problem(KP/TSP).')
    parser.add_argument('-cprob', '--crossoverProbability', dest='crossover_prob', action='store', 
                        default=0.7, type=float, help='Set the probability of crossover operator.')
    parser.add_argument('-mprob', '--mutationProbability', dest='mutation_prob', action='store',
                        default=0.001, type=float, help='Set the probability of mutation operator.')
    parser.add_argument('-psize', '--populationSize', dest='population_size', action='store',
                        default=300, type=int, help='Set the size of population.')
    parser.add_argument('-gen', '--generation', dest='generation', action='store',
                        default=500, type=int, help='Set the maximum generation.')
    parser.add_argument('-in', '--input', dest='filename', action='store',
                        default='tsp_data.txt', type=str, help='Set the input filename.')
    parser.add_argument('-out', '--output', dest='output', action='store',
                        default='result.json', type=str, help='Set the output filename(json).')
    args = parser.parse_args()
    # Check argument by standard output
    print(args)

    # Set arguments to variables
    problem = args.problem
    filename = args.filename
    crossover_prob = args.crossover_prob
    mutation_prob = args.mutation_prob
    popSize = args.population_size
    generation = args.generation

     # Traveling Salesman Problem Dataset
    dataset = Salesman(filename)
    dataset.read()
    gene_size = len(dataset.location)

    # Generate each GA instance
    tspga = GeneticAlgorithm(gene_size, popSize, crossover_prob, mutation_prob)

    best_fitness = []
    mean_fitness = []

    print('Ranking', end=' ')
    tspga.initialization()
    bf, mf = tspga.calculateFitness(dataset)
    best_fitness.append(bf)
    mean_fitness.append(mf)

    # Training
    for i in range(generation):
        # tspga.orderTwoCrossover()
        # tspga.cycleCrossover()
        tspga.partialMappedCrossover()

        # tspga.reorderMutation()
        tspga.offspringCalculateFitness(dataset)
        tspga.adaptiveReorderMutation()
        
        tspga.combination()
        bf, mf = tspga.calculateFitness(dataset)
        tspga.sortingSelection()

        best_fitness.append(bf)
        mean_fitness.append(mf)

    best_distance = []
    mean_distance = []
    for fit in best_fitness:
        best_distance.append(1/fit)
    for fit in mean_fitness:
        mean_distance.append(1/fit)

    # Print total best results.
    print('Ranking Selection best profit:', min(best_distance))
    print('Ranking Selection mean of profit:', min(mean_distance))

    x_type = 'pmx_cp-0.7_mp-0.001'

    # Save result to json format
    result_file = args.output
    if os.path.isfile(result_file):
        with open(result_file, 'r') as f:
            jsonDict = json.load(f)
        for i in range(generation+1):
            if x_type in jsonDict[i]:
                jsonDict[i][x_type].append(best_distance[i])
            else:
                jsonDict[i][x_type] = [best_distance[i]]
        jsonDict[-1]['best'].append(min(best_distance))
        jsonDict[-1]['mean'].append(min(mean_distance))
        jsonString = json.dumps(jsonDict, indent=4)
        with open(result_file, 'w') as f:
            f.write(jsonString)
        
    else:
        tmp_list = []
        for i in range(generation+1):
            tmp_dict = {
                'generation' : i,
                x_type : [best_distance[i]]
            }
            tmp_list.append(tmp_dict)
        tmp_best = {
            'best' : [min(best_distance)],
            'mean' : [min(mean_distance)]
        }
        tmp_list.append(tmp_best)
        jsonString = json.dumps(tmp_list, indent=4)
        with open(result_file, 'w') as f:
            f.write(jsonString)

    # Draw plot of results.
    plt.plot(range(generation+1), best_distance, color='green')
    plt.plot(range(generation+1), mean_distance, color='green', linestyle='--')
    plt.xlabel('generation')
    plt.ylabel('total profit')
    plt.legend([x_type+' best', x_type+' mean'])
    now = datetime.datetime.now()
    plt.savefig('result/' + now.strftime('%Y-%m-%d_%H:%M:%S') + '.png')
