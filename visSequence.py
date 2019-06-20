import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser(
        description='Visualize Key Points in CSV File')
parser.add_argument('--csv', default='pose.csv', help='CSV file where key points are read from')
args = parser.parse_args()


with open(args.csv, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')

	for row in reader:
		row_f = [float(i) for i in row]
		plt.scatter(row_f[0:36:2], [-i for i in row_f[1:36:2]], s=100, alpha=0.5)
	plt.show()
