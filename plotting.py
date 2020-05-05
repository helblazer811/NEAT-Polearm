import networkx as nx
import matplotlib.pyplot as plt
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
	