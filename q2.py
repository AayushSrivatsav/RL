import pandas as pd
import numpy as np
import seaborn as sns

#Problem statement: Find out which arm is the best arm to pull after n number of trials
#based on some heuristics and choose the best heuristic based on performance
#Trying to build an environment for this usecase with some assumptions
class MAB:
  def __init__(self, n):
    self.trials = np.zeros((n))        #Used to define number of trials
    self.arms = n                      #Used to define number of arms
    self.distribution = []             #Used to define distribution of rewards for each arm
    for i in range(n):
      self.distribution.append([np.random.randint(i,10), np.random.randint(i+10, 20)])
    print(f"Initialized MAB with {n} arms.")
    print(f"Distribution of rewards for each arm: Uniform in range{self.distribution}")

  def get_reward(self, arm):
    #Function to return reward for a particular arm.
    self.trials[arm] += 1
    return np.random.uniform(self.distribution[arm][0], self.distribution[arm][1])

  def get_trials(self):
    return self.trials

#Algorithm for Incremental Uniform
def IncrementalUniform(n = 10 ,total_trials=100):
  env = MAB(n)
  rewards = np.zeros((n))
  for i in range(total_trials):
    arm = np.argmin(env.get_trials())            #It's not random, change it to random
    reward = env.get_reward(arm)
    rewards[arm] = rewards[arm] + (reward - rewards[arm])/env.trials[arm]
    print(f"Arm {arm} was pulled with reward {reward}")
  print("Best arm: "+str(np.argmax(rewards)))
  return rewards

IncrementalUniform(2,10)


#Algorithm for greedy approach
def epsilonGreedy(epsilon = 0.75, n = 10, total_trials=100):
  env = MAB(n)
  rewards = np.zeros((n))
  for i in range(total_trials):
    if np.random.uniform(0,1) > epsilon:
      arm = np.argmin(env.get_trials())
    else:
      arm = np.argmax(rewards)
    reward = env.get_reward(arm)
    rewards[arm] = rewards[arm] + (reward - rewards[arm])/env.trials[arm]
    print(f"Arm {arm} was pulled with reward {reward}")
  print("Best arm: "+str(np.argmax(rewards)))
  return rewards

epsilonGreedy(0.75, 2, 10)

#Upper Confidence Bound
def UCB(n = 10, total_trials=100):
  env = MAB(n)
  rewards = np.zeros((n))
  for i in range(total_trials):
    if i < n:
      arm = i
    else:
      ucb_values = rewards + np.sqrt(2*np.log(sum(env.get_trials()))/env.get_trials())
      arm = np.argmax(ucb_values)
    reward = env.get_reward(arm)
    rewards[arm] = rewards[arm] + (reward - rewards[arm])/env.trials[arm]
    print(f"Arm {arm} was pulled with reward {reward}")
  print(f"Best arm: {np.argmax(rewards)}")
  return rewards

UCB(2,20)