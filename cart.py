
# IMPORTS
import gym
import numpy as np
import matplotlib.pyplot as plt
import random 
env = gym.make('CartPole-v0')

class NeuralNet:
    """
    Neural network to optimize the cartpole environment 
    """
    def __init__(self, input_dim, hidden_dim, output_dim, test_run):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.test_run = test_run

    #helper functions
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def init_weights(self):
        input_weight = []
        input_bias = []

        hidden_weight = []
        out_weight = []

        input_nodes = 4

        for i in range(self.test_run):
            inp_w = np.random.rand(self.input_dim, input_nodes)
            input_weight.append(inp_w)
            inp_b = np.random.rand((input_nodes))
            input_bias.append(inp_b)
            hid_w = np.random.rand(input_nodes, self.hidden_dim)
            hidden_weight.append(hid_w)
            out_w = np.random.rand(self.hidden_dim, self.output_dim)
            out_weight.append(out_w)

        return [input_weight, input_bias, hidden_weight, out_weight]

    def forward_prop(self, obs, input_w, input_b, hidden_w, out_w):

        obs = obs/max(np.max(np.linalg.norm(obs)), 1)
        Ain = self.relu(obs@input_w + np.array(input_b[0]).T)
        Ahid = self.relu(Ain@hidden_w)
        Zout = Ahid @ out_w
        A_out = self.relu(Zout)
        output = self.softmax(A_out)

        return np.argmax(output)

    def run_environment(self, input_w, input_b, hidden_w, out_w):
        obs = env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            action = self.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        return score

    def run_test(self):
        generation = self.init_weights()
        input_w, input_b, hidden_w, out_w = generation
        scores = []
        for ep in range(self.test_run):
            score = self.run_environment(
                input_w[ep], input_b[ep], hidden_w[ep], out_w[ep])
            scores.append(score)
        return [generation, scores]


class GA:
    """
    Training neural net using genetic algorithm
    """
    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.5):
        #initilizing different parameters of the GA in constructor
        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = []
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def crossover(self, DNA_list):
        """
        Generating number of offsprings from parents in DNA_list such that pop_size remains same. 
        """
        def cross(a,b,k):  #single point crossover
            newa=np.concatenate((a[:k],b[k:]))
            newb=np.concatenate((b[:k],a[k:]))
            return(newa,newb)        
       
        xr=[]
        newDNAs = []   #list to store offsprings
        dnaelement1=DNA_list[0]
        dnaelement2=DNA_list[1]
        # print(DNA_list)
        for i in range(self.population_size-2):
            x=random.choice(range(1,len(dnaelement1)))
            while x in xr:
                x=random.choice(range(1,len(dnaelement1)))
            xr.append(x)                            
            dnaelement1,dnaelement2=cross(dnaelement1,dnaelement2,x)   #implementing the crossing between parents
            newDNAs.append(dnaelement1)
            newDNAs.append(dnaelement2)
            i+=1
        if (self.population_size//2!=0):
            newDNAs.pop()
        return newDNAs

    def mutation(self, DNA):
        """
        Mutating DNA using mutation_rate to determine the mutation probability. 
        """

        for i in range(len(DNA)):
            if np.random.rand() < self.mutation_rate:
                x=random.randint(0,31)
                DNA[i][x] = np.random.rand()
         
        return DNA

    def next_generation(self):
        """
        Forms next generation from current generation.
        Before writing this function think of an appropriate representation of an individual in the population.
        Suggested method: Convert it into a 1-D array/list. This conversion is done for you in this function. Feel free to use any other method.
        Steps
        1. Crossover
        Suggested Method: select top two individuals with max fitness. generate remaining offsprings using these two individuals only.
        2. Mutation:
        """
        index_good_fitness = [] #index of parents selected for crossover.
        #fill the list.
        index_good_fitness = np.argsort(self.current_fitness)[::-1][:2]  #getting the index of parents with best best fitness
        new_DNA_list = []
        new_fitness_list = []

        DNA_list = []

        for index in index_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(w1.shape[1], -1)

            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)

            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(w2.shape[0], -1)
            dna_w2 = np.append(dna_b1, dna_whid)

            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh)
            DNA_list.append(dna)

        #parents selected for crossover moves to next generation
        new_DNA_list += DNA_list

        new_DNA_list += self.crossover(DNA_list)

        #mutate the new_DNA_list
        new_DNA_list = self.mutation(new_DNA_list)

        # converting 1D representation of individual back to original (required for forward pass of neural network)
        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:

            newdna_in_w1 = np.array(
                newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(
                newdna_in_w1, (-1, self.current_generation[0][0].shape[1]))
            new_input_weight.append(new_in_w)

            new_in_b = np.array(
                [newdna[newdna_in_w1.size:newdna_in_w1.size+self.current_generation[1][0].size]]).T  # bias
            new_input_bias.append(new_in_b)

            sh = newdna_in_w1.size + new_in_b.size
            newdna_in_w2 = np.array(
                [newdna[sh:sh+self.current_generation[2][0].size]])
            new_hid_w = np.reshape(
                newdna_in_w2, (-1, self.current_generation[2][0].shape[1]))
            new_hidden_weight.append(new_hid_w)

            sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
            new_out_w = np.array([newdna[sl:]]).T
            new_out_w = np.reshape(
                new_out_w, (-1, self.current_generation[3][0].shape[1]))
            new_output_weight.append(new_out_w)

            #evaluate fitness of new individual and add to new_fitness_list.
            #check run_environment function for details.
            fit=self.learner.run_environment(new_in_w,new_in_b,new_hid_w,new_out_w)
            new_fitness_list.append(fit)

        new_generation = [new_input_weight, new_input_bias,
                          new_hidden_weight, new_output_weight]
        return new_generation, new_fitness_list

    def show_fitness_graph(self):
        """
        Show the fitness graph
        Use fitness_list to plot the graph
        """
        plt.plot(self.fitness_list)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness over Generations')
        plt.show()

    def evolve(self):
        """
        Evolve the population
        Steps
        1. Iterate for number_of_generation and generate new population
        2. Find maximum fitness of an individual in this generation and update best_fitness 
        3. Append max_fitness to fitness_list
        4. Plot the fitness graph at end. Use show_fitnes_graph()
        """
        for generation in range(self.number_of_generation):
            self.current_generation, new_fitness_list = self.next_generation()
            max_fitness = max(new_fitness_list)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_gen = self.current_generation
            self.fitness_list.append(max_fitness)
        self.show_fitness_graph()
        #evolve
        return self.best_gen, self.best_fitness


def trainer():
    pop_size = 15
    num_of_generation = 100
    learner = NeuralNet(env.observation_space.shape[0], 2, env.action_space.n, pop_size)
    init_weight_list, init_fitness_list = learner.run_test()
    #instantiate the GA optimizer
    genetic=GA(init_weight_list,init_fitness_list,num_of_generation,pop_size,learner)
    #call evolve function to obtain optimized weights.
    bestgen,_=genetic.evolve()
    #return optimized weights
    return bestgen
    

def test_run_env(params):
    input_w, input_b, hidden_w, out_w = params
    obs = env.reset()
    score = 0
    learner = NeuralNet(env.observation_space.shape[0], 2, env.action_space.n, 15)
    for t in range(5000):
        env.render()
        action = learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
        obs, reward, done, info = env.step(action)
        score += reward
        print(f"time: {t}, fitness: {score}")
        if done:
            break
    print(f"Final score: {score}")

def main():
    params = trainer()
    test_run_env(params)


if(__name__ == "__main__"):
    main()