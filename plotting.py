import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""
	Set of helper functions for plotting
"""

"""
	Takes in a set of edges and draws the graph 
	using networkx 
"""
def draw_network(network, path="network.png"):
	nx.draw(network)

	labels = nx.get_edge_attributes(network,'weight')
	#shorten labels
	print(labels)
	for key in labels.keys():
		labels[key] = round(labels[key],3)
	pos=nx.planar_layout(network)
	nx.draw_networkx_edge_labels(network,pos, edge_labels=labels)
	plt.savefig(path)
	plt.show()
	

"""
	Plots a fitness histogram with different color
"""
def draw_histogram(data_frame, n_bins=10):
	rows = df.loc[df['generation'] == df['generation'].max()]
	species = rows['species'].tolist()
	fitness = rows['fitness'].tolist()

	species_fitnesses = [[] for i in range(max(species) + 1)]
	for i in range(len(fitness)):
		species_fitnesses[species[i]].append(fitness[i])


	plt.hist(species_fitnesses, n_bins, density=True, histtype='bar')
	plt.legend(prop={'size': 10})
	plt.show()

"""
	Plots the average fitness over the generations
"""
def plot_average_fitness(data_frame):
	average_fitness = []
	fitness_std = []
	upper_quartile = []
	lower_quartile = []
	gen_max = df['generation'].max()
	for i in range(gen_max):
		gen = df.loc[df['generation'] == i]
		fitnesses = gen['fitness']
		mean_fitness = np.median(fitnesses)
		std = np.std(fitnesses)
		upper_q = np.quantile(fitnesses, 0.75) 
		lower_q = np.quantile(fitnesses, 0.25)
		upper_quartile.append(upper_q)
		lower_quartile.append(lower_q)
		average_fitness.append(mean_fitness)
		fitness_std.append(std)

	plt.plot(average_fitness)
	average_fitness = np.array(average_fitness)
	plt.fill_between(np.arange(0,len(average_fitness)), lower_quartile, upper_quartile, color='cyan')
	plt.show()


"""
	Plots the average/mean fitnesses of each of the species
	To do this there needs to be continuity of species from generation to generation
"""
def plot_species_fitnesses(data_frame):
	pass

if __name__ == "__main__":
	# import the data
	df = pd.read_csv("data/data4.csv")
	#get rows with highest generation
	draw_histogram(df)
	plot_average_fitness(df)


