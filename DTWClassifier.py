import numpy as np
import fastdtw as fd
import argparse
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
from terminalPrintColors import tcolors 
    

folderStr = 'posesSubject1'
gestureExamples = ['posesSubject1/a1_s1_t2_color.avi.csv', 'posesSubject1/a6_s1_t3_color.avi.csv', 'posesSubject1/a9_s1_t2_color.avi.csv', 'posesSubject1/a24_s1_t4_color.avi.csv', 'posesSubject1/a7_s1_t1_color.avi.csv', 'posesSubject1/a26_s1_t4_color.avi.csv']

parser = argparse.ArgumentParser(
        description='Classify pose sequence in CSV')
parser.add_argument('--csvpath', '-cp', default='a1_', help='Path to CSV file to classify')
parser.add_argument('--resultcsvpath', '-rcp', default='result.csv', help='Path to CSV where result are written to')
parser.add_argument('--varthresh', '-vt', default='0.05', help='Min. variance to use dimension in DTW')
parser.add_argument('--sigma', '-s', default='0.5', help='Sigma for Gaussian smoothing')
args = parser.parse_args()

coordSequences1 = [[] for _ in range(36)] # two dim. list with list for every coordinate
normCoordSequences1 = [[] for _ in range(36)]
# read csv file of first sequence to list
with open(args.csvpath, 'r') as csvfile:
    reader_seq1 = csv.reader(csvfile, delimiter=',')
    for row in reader_seq1:
        row_f = [float(e) for e in row] # contains strings, cast to float explicitly
        for i in range(36): # 18 key points -> 36 coordinates
            coordSequences1[i].append(row_f[i])

print(tcolors.EMPH + args.csvpath + tcolors.ENDC)

minDistance = [-1, 999999.0]

for ge in gestureExamples:

    coordSequences2 = [[] for _ in range(36)]
    normCoordSequences2 = [[] for _ in range(36)]

    # read csv file of second sequence to list
    with open(ge, 'r') as csvfile:
        reader_seq2 = csv.reader(csvfile, delimiter=',')
        for row in reader_seq2:
            row_f = [float(e) for e in row] # contains strings, cast to float explicitly
            for i in range(36): # 18 key points -> 36 coordinates
                coordSequences2[i].append(row_f[i])

    compareDimensions = set() # add all dimensions to set that have a variance > threshold in either sequence

    for i,s in enumerate(coordSequences1):
        s = medfilt(s, int(args.sigma))
        varSeq = np.var(s)
        if varSeq > float(args.varthresh):
            compareDimensions.add(i)

    for i,s in enumerate(coordSequences2):
        s = medfilt(s, int(args.sigma))
        varSeq = np.var(s)
        if varSeq > float(args.varthresh):
            compareDimensions.add(i)

    ''' 
    For some reason, the commumative property is violated if the dimensions
    are in a different order... possibly rounding errors. Therefore they have
    to be sorted before performing the  DTW. 
    '''
    compareDimensions = sorted(compareDimensions)
    print(compareDimensions)

    distSum = 0.0

    for dim in compareDimensions:
        coordSequences1[dim] = gaussian_filter(coordSequences1[dim], 1)
        mean = np.mean(coordSequences1[dim])
        nmkps = [x-mean for x in coordSequences1[dim]]
        normCoordSequences1[dim] = nmkps

        mean = np.mean(coordSequences2[dim])
        coordSequences2[dim] = gaussian_filter(coordSequences2[dim], 1)
        nmkps = [x-mean for x in coordSequences2[dim]]
        normCoordSequences2[dim] = nmkps

        distance, path = fd.fastdtw(normCoordSequences1[dim], normCoordSequences2[dim], radius=30, dist=euclidean)
        distSum += distance
        #print(distance)

    distSum /= len(compareDimensions)
    print(tcolors.HEADER + "NORMALIZED DISTANCE: " + tcolors.ENDC + str(distSum) + ' : ' + ge)


    f = plt.figure()
    plt.subplot(2,1,2)
    for i in range(36):
        plt.plot(coordSequences1[i], 'xkcd:grey', label=i)

    for i in compareDimensions:
        plt.plot(normCoordSequences1[i], 'xkcd:black', label=i, linewidth=4.0)
        plt.plot(normCoordSequences1[i], label=i, linewidth=3.0)

    plt.subplot(2,1,1)
    for i in range(36):
        plt.plot(coordSequences2[i], 'xkcd:grey', label=i)

    for i in compareDimensions:
        plt.plot(normCoordSequences2[i], 'xkcd:black', label=i, linewidth=4.0)
        plt.plot(normCoordSequences2[i], label=i, linewidth=3.0)

    f.savefig('signals.pdf', bbox_inches='tight')
    
    if minDistance[1] > distSum:
        minDistance[0] = ge
        minDistance[1] = distSum

    #input('Press ENTER to continue...')

with open(args.resultcsvpath, mode='a') as res_csv_file:
    res_csv_file = csv.writer(res_csv_file, delimiter=',')
    res_csv_file.writerow((args.csvpath, minDistance[0]))
        
print('Classified as: ' + tcolors.EMPH + str(minDistance[0]) + tcolors.ENDC + ' (dist: ' + str(minDistance[1]) + ')')

print('\n')
