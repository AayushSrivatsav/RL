
#Importing libraries
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
regret = lambda x,n : max(x)*np.sum(n) - sum([x[i]*n[i] for i in range(len(x))])

class MAB:
  def __init__(self, n):
    self.trials = np.zeros((n))        #Used to define number of trials
    self.arms = n                      #Used to define number of arms
    self.ai = np.zeros((n))            #Used to define number of successes
    self.bi = np.zeros((n))            #Used to assign number of failures
    print(f"Initialized MAB with {n} arms with all ai and bi as 0...")

  def get_reward(self, arm, success):
    #Function to return reward for a particular arm.
    self.trials[arm] += 1
    self.pi = None
    if(self.ai[arm]>0 or self.bi[arm]>0):
      self.pi = self.ai[arm] / (self.ai[arm] + self.bi[arm])  #We don't know the population success probability so using sample success probability
    else:
      self.pi = 0
    if(success < self.pi):                                    #Based on success or failure, we are incrementing ai or bi
      self.bi[arm] += 1
    else:
      self.ai[arm] += 1
    return self.pi

  def get_trials(self):
    return self.trials

def decayingEpsilonGreedy(n=10, epsilon = 0.2, decay_rate = 0.02, total_trials = 100000):
  env = MAB(n)
  regret_array = np.zeros((total_trials))
  rewards = np.zeros((n))
  for i in range(total_trials):
    if(np.random.random() < epsilon):
      arm = np.random.randint(0,n)
    else:
      arm = np.argmax(rewards)
    success = np.random.random()        #For now we are making it random - need to get this data from user....
    if(i!=0):
      rewards[arm] = rewards[arm] +(1/i)*(env.get_reward(arm, success)-rewards[arm])
    else:
      rewards[arm] = env.get_reward(arm, success)
    regret_array[i] = regret(rewards, env.get_trials())
    epsilon = epsilon - epsilon * decay_rate
  print(f"Best arm after {total_trials} trials: ",np.argmax(rewards))
  #print(f"Regret after {total_trials} trials: ",regret_array)
  return regret_array

regrets = decayingEpsilonGreedy(n=100)
sns.lineplot(data = regrets)


def greedy(n = 10, total_trials = 100000):
  env = MAB(n)
  regret_array = np.zeros((total_trials))
  rewards = np.zeros((n))
  for i in range(total_trials):
    arm = None
    if(i<n):
      arm = i
    else:
      arm = np.argmax(rewards)
    success = np.random.random()
    if(i!=0):
      rewards[arm] = rewards[arm] +(1/i)*(env.get_reward(arm, success)-rewards[arm])
    else:
      rewards[arm] = env.get_reward(arm, success)
    regret_array[i] = regret(rewards, env.get_trials())
  print(f"Best arm after {total_trials} trials: ",np.argmax(rewards))
  #print(f"Regret after {total_trials} trials: ",regret_array)
  return regret_array

regrets_greedy = greedy(n=100)
sns.lineplot(data = regrets_greedy)

def UCB1(n = 10, total_trials = 100000, c = 2):
  env = MAB(n)
  regret_array = np.zeros((total_trials))
  rewards = np.zeros((n))
  for i in range(total_trials):
    if(i<n):
      arm = i
    else:
      arm = np.argmax(rewards + c*np.sqrt(np.log(i)/env.get_trials()))
    success = np.random.random()
    if(i!=0):
      rewards[arm] = rewards[arm] +(1/i)*(env.get_reward(arm, success)-rewards[arm])
    else:
      rewards[arm] = env.get_reward(arm, success)
    regret_array[i] = regret(rewards, env.get_trials())
  print(f"Best arm after {total_trials} trials: ",np.argmax(rewards))
  #print(f"Regret after {total_trials} trials: ",regret_array)
  return regret_array

regrets_UCB = UCB1(n=100)
sns.lineplot(data = regrets_UCB)
sns.lineplot(data = regrets_greedy)
sns.lineplot(data = regrets_UCB, color = "green")
plt.show()