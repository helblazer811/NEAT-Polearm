import gym
import numpy as np

# Helper functions

def relu(X):
   return np.maximum(0,X)


# Tests

"""
	Node object for a genome
"""
class Node():
	number = 0

	def __init__(self, node_type="hidden", activation=relu, output_number = None, input_number = None):
		self.num = Node.number + 1
		Node.number += 1
		self.type = node_type
		self.activation = activation #fixed for now
		self.incoming_edges = []
		self.outgoing_edges = []
		self.output_number = output_number
		self.input_number = input_number

	def __repr__(self):
		return "Node(number = "+str(self.num)+", type = "+str(self.type)+")"

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
		self.nodes = []
		self.connections = []
		input_nodes = []
		for i in range(input_size):
			new_node = Node(node_type = "input", input_number = i)
			input_nodes.append(new_node)

		output_nodes = []
		for i in range(output_size):
			new_node = Node(node_type = "output", output_number = i)
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
		Handles adding a connection to the graph
	"""
	def add_connection(self):
		pass

	"""
		Handles adding a node to the graph
	"""
	def add_node(self):
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

	def __init__(self, environment, genome = None):
		self.environment = environment
		self.genome = genome
		self.network = Network(environment.observation_space.shape[0],environment.action_space.n)

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
	def speciate(self, fitnesses, threshold = 1.5):
		# Calculate the distance metric of every genome in the population
		# with every other genome
		distances = []
		for r in range(self.population_size):
			row = []
			for c in range(0, self.population_size):
				if c == r:
					continue
				distance = Genome.distance(self.agents[r].network.genome, self.agents[c].network.genome)
				row.append(distance)
			distances.append(row)
		# Separate into species based on delta threshold
		removed = set()
		num_species = 0
		species = [-1 for i in range(self.population_size)]
		for i, agent in enumerate(self.agents):
			if agent in removed:
				continue
			#go through the distances relative to this agent
			#add species below a threshold of 1.5 to the same species
			num_species += 1
			species[i] = num_species
			removed.add(agent)
			for j, diff in enumerate(distances[i]):
				if j == i:
					continue
				if diff < threshold:
					species[j] = num_species
					removed.add(agents[j])

		return species


	"""
		Calculates the fitnesses for the genomes intra-species
	"""
	def calculate_mean_fitnesses(self, fitnesses, species):
		species_counts = [0] * max(species)
		for agent_species in species:
			species_counts[agent_species - 1] += 1

		mean_fitnesses = [0.0] * max(species) 
		for i, fitness in enumerate(fitnesses):
			this_species = species[i]
			mean_fitnesses[this_species - 1] += fitness / species_counts[this_species - 1]

		return mean_fitnesses

	"""
		Calculates the fitnesses of the species relative to eachother
	"""
	def calculate_intra_fitnesses(self, mean_fitnesses, species, fitnesses):
		pass

	"""
		Runs the Nuroevolution of Augmenting Topologies(NEAT) process
	"""
	def run_neuroevolution(self, n_generations, excess_c = 1, disjoint_c = 1, diff_threshold = 1):
		# Evaluates all agents in population
		fitnesses = self.evaluate() 
		# Separates the agents into species
		species = self.speciate(fitnesses) 
		# Calculate the species mean fitnesses
		# NOTE Ignored for now
		mean_fitnesses = self.calculate_mean_fitnesses(fitnesses, species) 
		# Calculate the intra species relative fitnesses
		# NOTE Ignored for now
		intra_fitnesses = self.calculate_intra_fitnesses(mean_fitnesses, species, fitnesses)
		# Select the genomes to survive based on the 
		# Kill the bottom third
		# Crossover the remaining ones randomly based on fitness values

	"""
		Returns the number of agents in the population
	"""
	def size(self):
		return len(self.agents)


# Initialize the environment
env = gym.make('CartPole-v0')
# Run the evolutionary process for n generations
population = Population(10, env)
neat_out = population.run_neuroevolution(50)

env.close()


# TODO
# Fix the distance metric
# Initialize the networks/genomes with the same topology
# Maybe employ clustering on the genome difference data
# Implement crossover
# Integrate random mutation
# Allow parameterization for mutation, crossover, species difference 
#      thresholds, at the top level in run_neuroevolution