import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotting
import networkx as nx

# Helper functions

def relu(X):
   return np.maximum(0,X)

# Tests

"""
	Node object for a genome
"""
class Node():
	number = 0

	def __init__(self, node_type="hidden", activation=relu, input_number=None, output_number=None):
		self.num = Node.number + 1
		Node.number += 1
		self.type = node_type
		self.activation = activation #fixed for now
		self.input_number = input_number
		self.output_number = output_number
		self.incoming_edges = []
		self.outgoing_edges = []

	def __repr__(self):
		return "Node(number = "+str(self.num)+", type = "+str(self.type)+")"

	"""
		Returns a copy of this object with empty incoming and leaving edges
	"""
	def copy(self):
		node_num = Node.number
		new_node = Node()
		new_node.type = self.type
		new_node.num = self.num
		new_node.input_number = self.input_number
		new_node.output_number = self.output_number
		new_node.activation = self.activation
		Node.number = node_num
		return new_node
"""
	Connection object for a genome
"""
class Connection():
	number = 0

	def __init__(self, in_node, out_node, weight=None, enabled=True):
		self.num = Connection.number + 1
		Connection.number += 1
		self.in_node = in_node
		self.out_node = out_node
		self.weight = weight if not weight == None else np.random.random_sample()
		self.enabled = enabled

	def __repr__(self):
		return "Connection(number = {}, input = {}, out = {})".format(self.num, self.in_node.num, self.out_node.num)

	"""
		Returns a copy of the connection without input/output nodes
	"""
	def copy(self):
		conn_num = Connection.number
		new_conn = Connection(None,None)
		new_conn.num = self.num
		new_conn.weight = self.weight
		new_conn.enabled = self.enabled
		Connection.number = conn_num
		return new_conn
"""
	The genome for an agent defining a network, 
	it contains all the information necessary to 
	build a network and perform the operations necessary
	for NEAT
"""
class Genome():
	
	def __init__(self):
		self.input_nodes = []
		self.output_nodes = []
		self.nodes = []
		self.connections = []

	"""
		Initializes a genome with no hidden units
		and connections randomized (-1,1) with ReLU
		activation
	"""
	def initialize_empty(self, input_size, output_size):
		#generate input and output nodes
		Node.number = 0
		Connection.number = 0

		self.nodes = []
		self.connections = []
		input_nodes = []
		for i in range(input_size):
			new_node = Node(node_type = "input", input_number=i)
			input_nodes.append(new_node)

		output_nodes = []
		for i in range(output_size):
			new_node = Node(node_type = "output", output_number=i)
			output_nodes.append(new_node)

		self.input_nodes = input_nodes
		self.output_nodes = output_nodes
		self.nodes.extend(input_nodes)
		self.nodes.extend(output_nodes)

		#create the connections
		connections = []
		for in_node in input_nodes:
			for out_node in output_nodes:
				connection = Connection(in_node, out_node)
				connections.append(connection)
				in_node.outgoing_edges.append(connection)
				out_node.incoming_edges.append(connection)
		self.connections = connections


	"""
		Mutates the given renome randomly given defined mutation rates
	"""
	def mutate(self, connection_rate=0.1, weight_rate=0.2):
		random_value = random.random()

		if random_value < connection_rate:
			self.add_random_connection()
		if random_value < weight_rate:
			self.change_random_weight()


	"""
		Adds a new random connection to the node
	"""
	def add_random_connection(self):
		# Take a random connection and add a node in between it
		# making one connection an identity and one the same as the 
		# previous weight
		rand_index = int(random.random() * len(self.connections))
		old_connection = self.connections[rand_index]
		old_connection.enabled = False
		new_node = Node()
		old_input = old_connection.in_node
		old_output = old_connection.out_node
		new_connection_a = Connection(old_input, new_node, weight=old_connection.weight)
		new_connection_b = Connection(new_node, old_output, weight=1.0) 
		new_node.incoming_edges.append(new_connection_a)
		new_node.outgoing_edges.append(new_connection_b)

	"""
		Changes a random weight
	"""
	def change_random_weight(self, sigma = 0.08):
		rand_index = int(random.random() * len(self.connections))
		# randomize a random weight
		old_weight = self.connections[rand_index].weight
		new_weight = np.random.normal(old_weight, sigma, 1)[0]
		new_weight = 0 if new_weight < 0.0 else new_weight
		new_weight = 1 if new_weight > 1.0 else new_weight

		self.connections[rand_index].weight = new_weight
		# TODO figure out if this is a good policy

	"""
		Crossover of two genomes
	"""
	@staticmethod
	def crossover(genome_a, genome_b):
		pass

	"""
		Measures the distance betweeen two genomes

		Excess : newer genes in one genome, but not the other
		Disjoint : old genes that dont exist in a genome
	"""
	@staticmethod
	def distance(genome_a, genome_b):
		distance = 0.0
		# TODO differentiate between disjoint and excess

		#make sets of the node and connection numbers for each genome
		a_connections = set([conn.num for conn in genome_a.connections])
		a_nodes = set([node.num for node in genome_a.nodes])
		b_connections = set([conn.num for conn in genome_b.connections])
		b_nodes = set([node.num for node in genome_b.nodes])

		#norm it with the bigger size between a and b's genomes
		max_connection_size = max(len(a_connections), len(b_connections))
		distance += len(a_connections.difference(b_connections)) / max_connection_size
		max_node_size = max(len(a_nodes), len(b_nodes))
		distance += len(a_nodes.difference(b_nodes)) / max_node_size

		#calculate weight differences for matching 
		diff = 0.0
		num_matching = 1
		for conn_a in genome_a.connections:
			for conn_b in genome_b.connections:
				if conn_a.num == conn_b.num:
					num_matching += 1
					diff += abs(conn_a.weight - conn_b.weight)
					break
		diff /= num_matching

		distance += diff

		return distance

	"""
		Deep copies the genome object
	"""
	def deepcopy(self):
		node_num = Node.number
		# create a one to one mapping of old nodes to new nodes
		node_map = {}
		for i, old_node in enumerate(self.nodes):
			node_map[old_node] = old_node.copy() # does not copy the incoming/outgoing edges
		# make a copy of the input_nodes
		input_nodes_copy = []
		for input_node in self.input_nodes:
			input_nodes_copy.append(node_map[input_node])
		# make a copy of the ouptut_nodes
		output_nodes_copy = []
		for out_node in self.output_nodes:
			output_nodes_copy.append(node_map[out_node])
		# create a one to one mapping of the old connections to the new
		conn_map = {}
		for i, old_conn in enumerate(self.connections):
			conn_map[old_conn] = old_conn.copy()
		# update the in/out nodes for each connection
		for old_conn in self.connections:
			conn_map[old_conn].in_node = node_map[old_conn.in_node]
			conn_map[old_conn].out_node = node_map[old_conn.out_node]
		# update the incoming/outgoing connections in each node
		for old_node in self.nodes:
			# incoming
			incoming = []
			for conn in old_node.incoming_edges:
				incoming.append(conn_map[conn])
			# outgoing
			outgoing = []
			for conn in old_node.outgoing_edges:
				outgoing.append(conn_map[conn])

			node_map[old_node].outgoing_edges = outgoing
			node_map[old_node].incoming_edges = incoming
		# return the genome
		copied_genome = Genome()
		copied_genome.input_nodes = input_nodes_copy
		copied_genome.output_nodes = output_nodes_copy
		copied_genome.nodes = list(node_map.values())
		copied_genome.connections = list(conn_map.values())
		return copied_genome

	"""
		Convert genome to a networkx format
	"""
	def convert_to_networkx(self):
		network = nx.Graph()

		converted_nodes = [node.num for node in self.nodes]
		network.add_nodes_from(converted_nodes)

		for conn in self.connections:
			network.add_edge(conn.in_node.num, conn.out_node.num, weight=conn.weight)

		return network


"""
	A neural network that is being used as the best action 
	prediction function for the given environment task

"""
class Network():
	
	def __init__(self, input_size, output_size, genome = None):
		if genome != None:
			self.genome = genome
		else:
			self.genome = Genome()
			self.genome.initialize_empty(input_size, output_size)

		self.input_size = input_size
		self.output_size = output_size
		self.sorted_genome = Network.topological_sort(self.genome)

	"""
		Network that takes in the input state and outputs
		a prediction of the best action
	"""
	def predict(self,state):
		values = {}
		outputs = [0] * len(self.genome.output_nodes)
		def compute_node_output(node):
			total = 0.0
			for pred_edge in node.incoming_edges:
				weight = pred_edge.weight
				value = values[pred_edge.in_node.num]
				total += weight * value
			return node.activation(total)

		#set input values for the nodes
		for i, input_node in enumerate(self.genome.input_nodes):
			values[input_node.num] = state[i]
		for i, node in enumerate(self.sorted_genome):
			if node.type == "input":
				continue
			values[node.num] = compute_node_output(node)
			if node.type == "output":
				outputs[node.output_number] = values[node.num]

		return np.argmax(outputs)

	"""
		Does topoligical sort of the genome connections
		so only a single pass of the graph is necessary
	"""
	@staticmethod
	def topological_sort(genome):
		#NOTE assumes a DAG is given, this must be taken care of in the mutators
		no_incoming = []
		no_incoming.extend(genome.input_nodes)
		removed_edges = set()
		#kahns algorithm
		sorted_nodes = []
		while len(no_incoming) > 0:
			current_node = no_incoming.pop(0)
			sorted_nodes.append(current_node)
			for edge in current_node.outgoing_edges:
				removed_edges.add(edge)
				empty = True
				for next_edge in edge.out_node.incoming_edges:
					if not next_edge in removed_edges:
						empty = False
						break
				if empty:
					no_incoming.append(edge.out_node)
		return sorted_nodes

"""
	Agent that belongs to the population used in the 
	evolutionary process. It holds a network as well as 
	additional information required to run the evolutionary 
	process. 
"""
class Agent():

	def __init__(self, environment, genome=None):
		self.environment = environment
		self.network = Network(environment.observation_space.shape[0],environment.action_space.n, genome=genome)

	"""
		Evaluates the fitness of the agent based on the passed environment
	"""
	def evaluate(self):
		fitness = 0.0

		observation = env.reset()
		for t in range(100):
			env.render()
			action = self.network.predict(observation)
			observation, reward, done, info = env.step(action)
			if done:
				fitness = t * 1.0
				break

		return fitness

	"""
		Makes a copy of this agent
	"""
	def deepcopy(self):
		copied_genome = self.network.genome.deepcopy()
		copied_agent = Agent(self.environment, genome=copied_genome)
		return copied_agent

"""
	A population of Agents meant to solve the task 
	described by the environment
"""
class Population():

	def __init__(self, population_size, environment):
		self.agents = [Agent(environment) for i in range(population_size)]
		self.population_size = population_size

	"""
		Evaluates the fitness of every agent in the population
	"""
	def evaluate(self):
		fitnesses = []
		for agent in self.agents:
			agent_fitness = agent.evaluate()
			fitnesses.append(agent_fitness)

		return fitnesses

	"""
		Divides the population into several species based on 
		the genome 
	"""
	def speciate(self, fitnesses, threshold = 0.3):
		# Calculate the distance metric of every genome in the population
		# with every other genome
		distances = []
		for r in range(self.population_size):
			row = []
			for c in range(0, self.population_size):
				if c == r:
					row.append(0.0)
					continue
				distance = Genome.distance(self.agents[r].network.genome, self.agents[c].network.genome)
				row.append(distance)
			distances.append(row)
		# Separate into species based on delta threshold
		removed = set()
		num_species = 0
		species = [-1 for i in range(self.population_size)]
		for i, agent in enumerate(self.agents):
			if len(removed) == len(fitnesses):
				break
			if agent in removed:
				continue
			#go through the distances relative to this agent
			#add species below a threshold of 1.5 to the same species
			species[i] = num_species
			removed.add(agent)
			for j, diff in enumerate(distances[i]):
				if j == i:
					continue
				if diff < threshold:
					species[j] = num_species
					removed.add(self.agents[j])

			num_species += 1

		return species

	"""
		Calculates the fitnesses for the genomes intra-species
	"""
	def calculate_mean_fitnesses(self, fitnesses, species):
		species_counts = [0] * (max(species) + 1)
		for agent_species in species:
			species_counts[agent_species] += 1

		mean_fitnesses = [0.0] * (max(species) + 1)
		for i, fitness in enumerate(fitnesses):
			this_species = species[i]
			mean_fitnesses[this_species] += fitness / species_counts[this_species]

		return mean_fitnesses

	"""
		Calculates the fitnesses of the species relative to eachother
	"""
	def calculate_intra_fitnesses(self, mean_fitnesses, species, fitnesses):
		relative_fitnesses = [fitnesses[i] - mean_fitnesses[species[i]] for i in range(len(species))]

		num_species = max(species) + 1
		# compress between zero and one for each species min/max
		species_mins = [None] * num_species
		species_maxs = [None] * num_species
		for i,fitness in enumerate(relative_fitnesses):
			if species_mins[species[i]] == None or fitness < species_mins[species[i]]:
				species_mins[species[i]] = fitness
			if species_maxs[species[i]] == None or fitness > species_maxs[species[i]]:
				species_maxs[species[i]] = fitness

		for i in range(len(relative_fitnesses)):
			species_range = (species_maxs[species[i]] - species_mins[species[i]])
			bottom_value = -species_mins[species[i]]
			if species_range == 0:
				relative_fitnesses[i] = 0.0
			else:
				relative_fitnesses[i] = (relative_fitnesses[i] + bottom_value) / species_range

		return relative_fitnesses

	"""
		Returns a list of surviving agents based on randomness weighted by the 
		intra-species relative fitnesses
	"""
	def select_survivors(self, intra_species_fitnesses, species, survival_rate=0.7):
		# If the normalized fitness is smaller then 0.5 there is a 30% chance of
		# dropping it
		survivors = [1] * len(intra_species_fitnesses)
		uniform = np.random.uniform(0, 1, len(intra_species_fitnesses))
		for i, fitness in enumerate(intra_species_fitnesses):
			if intra_species_fitnesses[i] < 0.5 and uniform[i] > survival_rate:
				survivors[i] = 0

		return survivors

	"""
		Generates a new population based on crossover and mutation of the previous
		generation
	"""
	def next_generation(self, survivors, species):
		# TODO implement crossover
		# Randomly copy the survivors until the population is filled
		new_agents = []
		for i, survivor in enumerate(survivors):
			if survivor == 1:
				new_agents.append(self.agents[i])

		while len(new_agents) < self.population_size:
			rand_int = int(random.random() * len(new_agents))
			copied_agent = self.agents[rand_int].deepcopy()
			new_agents.append(copied_agent)
		# mutate all of them
		for i, agent in enumerate(new_agents):
			agent.network.genome.mutate()

		return new_agents

	"""
		Adds data about all agents of this generation to to the dataframe
	"""
	def add_generation_to_data_array(self, data_array, fitnesses, species, mean_fitnesses, survivors):
		for agent_index in range(len(fitnesses)):
			row = {
				"fitness": fitnesses[agent_index],
				"species": species[agent_index],
				"survived": survivors[agent_index] == 1,
				"mean_fitness": mean_fitnesses[species[agent_index]]
			}
			data_array.append(row)

	"""
		Runs the Nuroevolution of Augmenting Topologies(NEAT) process
	"""
	def run_neuroevolution(self, n_generations, excess_c = 1, disjoint_c = 1, diff_threshold = 1, data_array=None, snapshots=None):
		# Iterate for n_generations
		for generation in range(0, n_generations):
			# Evaluates all agents in population
			fitnesses = self.evaluate()
			# Separates the agents into species
			species = self.speciate(fitnesses)
			# Calculate the species mean fitnesses
			mean_fitnesses = self.calculate_mean_fitnesses(fitnesses, species)
			# Calculate the intra species relative fitnesses
			intra_fitnesses = self.calculate_intra_fitnesses(mean_fitnesses, species, fitnesses)
			# Select a list of agents to survive
			surviving_agents = self.select_survivors(intra_fitnesses, species)
			# Crossover/ mutate the survivors
			next_generation = self.next_generation(surviving_agents, species)
			# Re-assign agents to this next generation
			self.agents = next_generation
			# Append to dataframe
			if not data_array is None:
				self.add_generation_to_data_array(data_array, fitnesses, species, mean_fitnesses, surviving_agents)

			#plotting.draw_network(self.agents[0].network.genome.convert_to_networkx())

	"""
		Returns the number of agents in the population
	"""
	def size(self):
		return len(self.agents)

# Plotting functions 

"""
	Plots the fitnesses of the species
"""
def plot_fitness(species, fitnesses):
	num_species = max(species) + 1
	species_fitnesses = {}
	for i in range(len(species)):
		if not species[i] in species_fitnesses.keys():
			species_fitnesses[species[i]] = [fitnesses[i]]
		species_fitnesses[species[i]].append(fitnesses[i])

	#print(species_fitnesses)
	#turn data into pandas dataframe
	#df = pd.DataFrame(species_fitnesses)
	
	#ax = sns.violinplot(data=df, palette="muted")

	#plt.draw()
# Plotting settings

# Initialize the environment
env = gym.make('CartPole-v0')
# Create population
population = Population(5, env)
# Create storage arrays
data_array = []
snapshots = [] 
# Run the evolutionary process for n generations
neat_out = population.run_neuroevolution(5, data_array=data_array, snapshots=snapshots)
# Close environment 
env.close()
# Create dataframe
df = pd.DataFrame(columns=['species','fitness','survived','mean_fitness']) #data frame that holds the results of 
df = df.append(data_array)
# Pickle data frame
df.to_csv("data.csv")


# TODO
# Fix the distance metric
# Maybe employ clustering on the genome difference data
# Implement crossover
# Make a better selection rule
# Integrate random mutation
# Allow parameterization for mutation, crossover, species difference 
#      thresholds, at the top level in run_neuroevolution