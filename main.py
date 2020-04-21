import gym
import numpy as np

# Helper functions

def relu(X):
   return np.maximum(0,X)

"""
	Node object for a genome
"""
class Node():

	def __init__(self, num, node_type="hidden", activation=relu):
		self.num = num
		self.type = node_type
		self.activation = activation #fixed for now
		self.incoming_edges = []
		self.outgoing_edges = []

"""
	Connection object for a genome
"""
class Connection():

	def __init__(self, in_node, out_node, weight=None, enabled=True):
		self.in_node = in_node
		self.out_node = out_node
		self.weight = weight if not weight == None else np.random.random_sample()
		self.enabled = enabled

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
		num_nodes = 0
		input_nodes = []
		for i in range(input_size):
			new_node = Node(num_nodes, node_type = "input")
			input_nodes.append(node)
			num_nodes += 1

		output_nodes = []
		for i in range(output_size):
			new_node = Node(num_nodes, node_type = "output")
			output_nodes.append(node)
			num_nodes +=1
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
	A neural network that is being used as the best action 
	prediction function for the given environment task

"""
class Network():
	
	def __init__(self, genome, input_size, output_size):
		self.genome = genome
		self.input_size = input_size
		self.output_size = output_size
		self.predict_func = self.build_predict()

	"""
		Network that takes in the input state 
	"""
	def predict(self, state):
		#runs the prediction function on the input state
		return self.predict_func(state)

	"""
		Builds a function that computes the outputs of the graph
	"""
	def build_predict(self):
		sorted_genome = Network.topological_sort(self.genome)

		def predict_func(state):
			values = {}
			outputs = {}
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

			for i, node in sorted_genome:
				if node.type == "input":
					continue
				values[node.num] = compute_node_output(node)
				if node.type == "output":
					outputs[node.num] = values[node.num]

			return outputs
	
		return predict_func

	"""
		Does topoligical sort of the genome connections
		so only a single pass of the graph is necessary
	"""
	@classmethod
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
      			removed_edges.append(edge)
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

	def __init__(self, environment, genome):
		self.environment = environment
		self.genome = genome
		self.network = Network(genome, environment.action_space.n, environment.input_space.n)

	"""
		Evaluates the fitness of the agent based on the passed environment
	"""
	def evaluate(self):
		fitness = 0.0

		return fitness

"""
	A population of Agents meant to solve the task 
	described by the environment
"""
class Population():

	def __init__(self, population_size, environment):
		self.agents = [Agent(environment) for i in range(population_size)]

	"""
		Returns the number of agents in the population
	"""
	def size(self):
		return len(self.agents)

"""
	Runs the NEAT process for n_iterations
"""
def run_neat(n_iterations, environment, population_size=100):
	population = Population(population_size, environment)

	for iteration in range(n_iterations):
		pass


# Initialize the environment
env = gym.make('CartPole-v0')
# Run the evolutionary process for n generations
neat_out = run_neat(50, env)