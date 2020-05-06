import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
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
def draw_histogram(fitnesses, species, n_bins=10):
	species_fitnesses = [[] for i in range(max(species) + 1)]
	for i in range(len(fitnesses)):
		species_fitnesses[species[i]].append(fitnesses[i])


	plt.hist(species_fitnesses, n_bins, density=True, histtype='bar')
	plt.legend(prop={'size': 10})
	plt.show()
