import gym
import neat
import pandas as pd

# Initialize the environment
env = gym.make('CartPole-v0')
# Create population
population = neat.Population(80, env)
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
df.to_csv("data.csv")
#get rows with highest generation
rows = df.loc[df['generation'] == df['generation'].max()]
species = rows['species'].tolist()
fitness = rows['fitness'].tolist()
plotting.draw_histogram(fitness,species)
# Pickle data frame
