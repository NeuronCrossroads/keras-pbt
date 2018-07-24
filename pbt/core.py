import time
import numpy as np
from copy import deepcopy as dc

class Worker:
    '''
    This is the basic class. Essentially an individual in the population that will execute config.
    Requirements:
     - A Neural Network Model CLASS
     - Training Function A FUNCTION
     - Evaluation Function FUNCTION
     - Config Dict
    
    Returns:
     - New Config Dict
     - Training Losses List
     - Score for objective
    
    Each of these have their own requirements:
    Neural Network Model:
     - Must be a Pytorch nn.Module
     - Uninstantiated(Must be the class)
     - Hyperparameters must be changeable after initialization(like lr, and dropout p)
       using a set_parameters method for the model class
    
    
    Train Function:
     - Must input model(positional, 1st argument) and its config dictionary
     - Must output: final loss at the end generation
    
    
    Evaluate Function:
     - Must input model and other required params(eg. episodes,data)
     - must output: objective score as python float
    
    
    Config Dict:
     - Must be a dictionary of dictionaries
     - Contain:
      - The generation(starting with 0) as key 'generation'
    
     - Contain arguments for model,trainer,and evaluator so they can be called like so:
      + model(**config['hyperparameters'])
      + trainer(model,**config['trainer'])
       - By default, the trainer config will not change between generations
      + evaluator(model,**config[evaluator])
       - By default, the evaluator config will not change between generations
    '''
    def __init__(self,model,trainer,evaluator,config):
        self.model = model(**config['hyperparameters'])
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator

    def train(self):
        self.losses = self.trainer(self.model,**self.config['trainer'])
    
    def evaluate(self):
        self.score = self.evaluator(self.model,**self.config['evaluator'])
    
    def run(self):
        self.train()
        self.evaluate()
        return self.losses, self.score

class PBTTrainer:
    '''
    Population Based Trainer(Synchronous):
    '''
    def __init__(self,population_size,generations,
                 model,trainer,evaluator, 
                 rand_scales, config, explore=(1.2,0.8), rank='high'):
        self.pop_size = population_size
        self.generations = generations
        self.generation = 0
        self.rank = rank
        self.explore = explore
        self.config = config
        self.model = model
        self.trainer = trainer
        self.evaluator = evaluator
        
        self.losses = []
        self.scores = []
        
        self.population = []
        for individual in range(self.pop_size):
            new_config = dc(config)
            #Randomize parameters at the beginning
            for param,value in config['hyperparameters'].items():
                new_config['hyperparameters'][param] = np.random.uniform(1/rand_scales[param],rand_scales[param])*value
            self.population.append(Worker(model,trainer,evaluator,new_config))
        
    
    def run_generation(self):
        self.losses.append([])
        self.scores.append([])
        for i,worker in enumerate(self.population):
            loss, score = worker.run()
            self.losses[-1].append(loss)
            self.scores[-1].append(score)
            print ('\rFinished I',i+1,end='')
        self.generation += 1
        
    
    def ranker(self):
        scores = self.scores[-1]
        #If objective is to get highest score metric
        if self.rank == 'high':
            scores = np.multiply(scores,-1).tolist()
        ranked = sorted(range(len(scores)), key=lambda k: scores[k])
        return ranked

    def exploit_explore(self):
        #EXPLOIT/EXPLOIT: Truncation Selection/Perturb
        ranks = self.ranker()#Gives indicies from best to worst score
        num = int(np.ceil(0.2*self.pop_size))
        best = ranks[:num]
        worst = ranks[-num:]
        for idx in range(num):
            self.population[worst[idx]].model.model.set_weights(self.population[best[idx]].model.model.get_weights())
            
            new_hyperparameters = dc(self.population[best[idx]].config['hyperparameters'])
            if np.random.uniform() < 0.5:
                factor = self.explore[0]
            else:
                factor = self.explore[1]
            for param,value in new_hyperparameters.items():
                    new_hyperparameters[param] = value * factor
            self.population[worst[idx]].model.set_hyperparameters(**new_hyperparameters)
    
    def run(self):
        speed = 0
        for i in range(self.generations):
            tic = time.time()
            self.run_generation()
            self.exploit_explore()
            toc = time.time()
            speed = speed*i/self.generations + (toc-tic)*(1-i/self.generations)
            metrics = [i+1,np.max(self.scores[-1]),speed,speed*self.generations-speed*(i+1)]
            print ('\nG {}: Best Score so Far: {:.2f}\tTime per generation: {:.2f} secs\tTime Remaining: {:.2f} secs'.format(*metrics))