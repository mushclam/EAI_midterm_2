import os
import json
import numpy as np
import datetime
import argparse
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    os.chdir(path)

    parser = argparse.ArgumentParser(description='Output json file encoder')
    parser.add_argument('-in', '--input', dest='input', action='store',
                        default='result.json', type=str, help='Set the input filename.')
    parser.add_argument('-out', '--out', dest='output', action='store',
                        default='total_result.png', type=str, help='Set the output filename.')
    args = parser.parse_args()
    print(args)

    json_file = args.input
    with open(json_file, 'r') as f:
        jsonDict = json.load(f)

    generations = []
    keys = list(jsonDict[0].keys())
    keys.remove('generation')

    means = {}
    stds = {}
    for element in jsonDict:
        generations.append(element['generation'])
        for key in keys:
            ga_result = np.array(element[key])
            if not key in means:
                means[key] = [ga_result.mean()]
            else:
                means[key].append(ga_result.mean())
            if not key in stds:
                stds[key] = [ga_result.std()]
            else:
                stds[key].append(ga_result.std())

    for key in keys:
        plt.errorbar(generations, means[key], stds[key])
    plt.xlabel('generation')
    plt.ylabel('total profit')
    plt.legend([k for k in keys])
    now = datetime.datetime.now()
    plt.savefig('result/' + args.output)