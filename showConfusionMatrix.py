import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import pandas as pd
import seaborn as sn
from terminalPrintColors import tcolors 

parser = argparse.ArgumentParser(
        description='Show Confusion Matrix of Classification')
parser.add_argument('--csvpath', '-cp', default='result.csv', help='Path to CSV file with results')
args = parser.parse_args()

confMat = [[0,0,0,0,0,0] for _ in range(6)] # TODO: this is hard-coded to 6 different gestures
#confMat = [[0,0,0,0,0,0,0] for _ in range(7)] # TODO: this is hard-coded to 7 different gestures


#f = plt.figure()

with open(args.csvpath, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    n = 0
    c = 0
    for row in reader:
        n += 1
        groundTruthClass = str(row[0])
        classifiedAs = str(row[1])
        i = -1
        j = -1

        
        # TODO: automatically adjust to number of gestures being classified
        '''
        if groundTruthClass.startswith('posesSubject1/a1_'):
            i = 0
        if groundTruthClass.startswith('posesSubject1/a6_'):
            i = 1
        if groundTruthClass.startswith('posesSubject1/a7_'):
            i = 2
        if groundTruthClass.startswith('posesSubject1/a8_'):
            i = 3
        if groundTruthClass.startswith('posesSubject1/a9_'):
            i = 4
        if groundTruthClass.startswith('posesSubject1/a24_'):
            i = 5
        if groundTruthClass.startswith('posesSubject1/a26_'):
            i = 6

        if classifiedAs.startswith('poseCSVs/a1_'):
            j = 0
        if classifiedAs.startswith('poseCSVs/a6_'):
            j = 1
        if classifiedAs.startswith('poseCSVs/a7_'):
            j = 2
        if classifiedAs.startswith('poseCSVs/a8_'):
            j = 3
        if classifiedAs.startswith('poseCSVs/a9_'):
            j = 4
        if classifiedAs.startswith('poseCSVs/a24_'):
            j = 5
        if classifiedAs.startswith('poseCSVs/a26_'):
            j = 6
        '''
        if groundTruthClass.startswith('poseCSVs/a1_'):
            i = 0
        if groundTruthClass.startswith('poseCSVs/a6_'):
            i = 1
        if groundTruthClass.startswith('poseCSVs/a7_'):
            i = 2
        if groundTruthClass.startswith('poseCSVs/a9_'):
            i = 3
        if groundTruthClass.startswith('poseCSVs/a24_'):
            i = 4
        if groundTruthClass.startswith('poseCSVs/a26_'):
            i = 5

        if classifiedAs.startswith('posesSubject1/a1_'):
            j = 0
        if classifiedAs.startswith('posesSubject1/a6_'):
            j = 1
        if classifiedAs.startswith('posesSubject1/a7_'):
            j = 2
        if classifiedAs.startswith('posesSubject1/a9_'):
            j = 3
        if classifiedAs.startswith('posesSubject1/a24_'):
            j = 4
        if classifiedAs.startswith('posesSubject1/a26_'):
            j = 5
        
        if i == j:
            c += 1


        confMat[j][i] += 1


#df_cm = pd.DataFrame(confMat, index = [i for i in ['a1', 'a6', 'a7', 'a8', 'a9', 'a24', 'a26']], columns =[i for i in ['a1', 'a6', 'a7', 'a8', 'a9', 'a24', 'a26']])
df_cm = pd.DataFrame(confMat, index = [i for i in ['a1', 'a6', 'a7', 'a9', 'a24', 'a26']], columns =[i for i in ['a1', 'a6', 'a7', 'a9', 'a24', 'a26']])
plt.figure(figsize=(10,7))


sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, cmap='gray_r')

print(c,n)
print('Correctly Classified: ' + tcolors.ERR + str(100 * float(c)/n) + ' %' + tcolors.ENDC)
#plt.matshow(confMat)
plt.draw()
plt.savefig('confMat.pdf', bbox_inches='tight')
