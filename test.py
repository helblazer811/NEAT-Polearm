import gym
import neat
import pandas as pd
import os, os.path

def get_unique_name():
	size = len(os.listdir('data'))
	return "data"+str(size)+".csv"

# Initialize the environment
env = gym.make('CartPole-v0')
# Create population
population = neat.Population(30, env)
# Create storage arrays
data_array = []
snapshots = [] 
# Run the evolutionary process for n generations
neat_out = population.run_neuroevolution(30, data_array=data_array, snapshots=snapshots)
# Close environment 
env.close()
# Create dataframe
df = pd.DataFrame(columns=['generation','species','fitness','survived','mean_fitness']) #data frame that holds the results of 
df = df.append(data_array)

df.to_csv("data/"+get_unique_name())
