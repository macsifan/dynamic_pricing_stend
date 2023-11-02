import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def simulate_user_behaviour_1(price_points):
    user_demand_data = []

    for price in price_points:
        user_ids = np.arange(1, self.user_count + 1)
        demand_prob = self.high_demand_users_ratio / self.max_demand
        demands = np.random.choice(
            [self.min_demand, *range(1, self.max_demand + 1)],
            self.user_count,
            p=[1 - self.high_demand_users_ratio, *(demand_prob for _ in range(self.max_demand))]
        )
        adjusted_demand = self.demand(price)
        demands = (demands * adjusted_demand * 4.5).astype(int)

        data = pd.DataFrame({
            'user': user_ids,
            'demand': demands,
            'price': price,
            'revenue': price * demands
        })

        user_demand_data.append(data)

    return pd.concat(user_demand_data)


# In[3]:


class DemandSimulator:
    def __init__(self, user_count=1000, demand_exponent=-0.00175, max_demand=10, min_demand=0, high_demand_users_ratio=0.1):
        self.user_count = user_count
        self.demand_exponent = demand_exponent
        self.max_demand = max_demand
        self.min_demand = min_demand
        self.high_demand_users_ratio = high_demand_users_ratio

    def demand(self, price):
        demand = np.exp(self.demand_exponent * price)
        return demand / (1 + demand)

    def simulate_user_behaviour_1(self, price_points):
        user_demand_data = []

        for price in price_points:
            user_ids = np.arange(1, self.user_count + 1)
            demand_prob = self.high_demand_users_ratio / self.max_demand
            demands = np.random.choice(
                [self.min_demand, *range(1, self.max_demand + 1)],
                self.user_count,
                p=[1 - self.high_demand_users_ratio, *(demand_prob for _ in range(self.max_demand))]
            )
            adjusted_demand = self.demand(price)
            demands = (demands * adjusted_demand * 4.5).astype(int)

            data = pd.DataFrame({
                'user': user_ids,
                'demand': demands,
                'price': price,
                'revenue': price * demands
            })

            user_demand_data.append(data)

        return pd.concat(user_demand_data)

    def simulate_user_behaviour_2(self, price_points):
        user_demand_data = []

        for price in price_points:
            user_ids = np.arange(1, self.user_count + 1)
            demand_prob = self.high_demand_users_ratio / self.max_demand
            demands = np.random.choice(
                [self.min_demand, *range(1, self.max_demand + 1)],
                self.user_count,
                p=[1 - self.high_demand_users_ratio, *(demand_prob for _ in range(self.max_demand))]
            )
            adjusted_demand = self.demand(price)
            if price <= 800:
                demands = (demands * adjusted_demand * 1).astype(int)
            else:
                demands = (demands * adjusted_demand * 1.4).astype(int)

            data = pd.DataFrame({
                'user': user_ids,
                'demand': demands,
                'price': price
            })

            user_demand_data.append(data)
        user_demand_data = pd.concat(user_demand_data)
        user_demand_data.loc[:, 'demand'] = user_demand_data.loc[:, 'demand'].values[::-1]
        user_demand_data['revenue'] = user_demand_data['price'] * user_demand_data['demand']
        return user_demand_data
        #return pd.concat(user_demand_data)

    def plot_demand_and_revenue(self, df):
        to_pl = df.groupby(['price']).agg({'revenue': 'sum', 'demand': 'sum'}).reset_index()

        optimal_point = to_pl.revenue.argmax()
        optimal_price = to_pl.iloc[optimal_point].price
        optimal_revenue = to_pl.iloc[optimal_point].revenue

        print(f"Optimal Price: {optimal_price}")
        print(f"Maximum Revenue: {optimal_revenue}")

        plt.figure(figsize=(10, 6))
        plt.plot(to_pl['price'], to_pl['demand'], marker='o', linestyle='-')
        plt.xlabel('Цена', fontsize = 'xx-large')
        plt.ylabel('Спрос', fontsize = 'xx-large')
        plt.title('График кривой спроса')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(to_pl['price'], to_pl['revenue'], marker='o', linestyle='-')
        plt.xlabel('Цена', fontsize = 'xx-large')
        plt.ylabel('Выручка', fontsize = 'xx-large')
        plt.title('Оптимум')
        plt.grid(True)
        plt.show()
        
    def plot_demand_products(self, df):
        plt.figure(figsize=(10, 6))
        
        df['revenue'] = df['revenue_1'] + df['revenue_2']        
        to_pl = df.groupby(['price_1']).agg({'revenue': 'sum'}).reset_index()
        optimal_point = to_pl.revenue.argmax()
        optimal_price = to_pl.iloc[optimal_point].price_1
        optimal_revenue = to_pl.iloc[optimal_point].revenue

        print(f"Optimal Price: {optimal_price}")
        print(f"Maximum Revenue: {optimal_revenue}")
        

        plt.plot(to_pl.price_1, to_pl.revenue, label = 'Product #1', marker = 'o', linestyle = '-')
        plt.xlabel('Price', fontsize = 'xx-large')
        plt.ylabel('Revenue', fontsize = 'xx-large')
        plt.grid(True)
        plt.legend();


# In[4]:


class EpsGreedy():
    def __init__(self, price_points, eps: float = 0.1):
        self.price_points = price_points
        self.n_arms = len(self.price_points)
        self.n_iters = 0
        self.arms_states = np.zeros(self.n_arms)
        self.arms_actions = np.zeros(self.n_arms)
        self.eps = eps
        
    def select_price(self):
        if random.random() < self.eps:
            a = np.zeros(self.n_arms)
            a[random.randint(0, self.n_arms - 1)] = 1
            
            return a
        else:
            return self.arms_states / (self.arms_actions+ 1e-5)
        
    def update(self, arm: int, reward: int):
        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1
        
        
        
class UCB():
    def __init__(self, price_points):
        self.price_points = price_points
        self.n_arms = len(self.price_points)
        self.n_iters = 0
        self.arms_states = np.zeros(self.n_arms)
        self.arms_actions = np.zeros(self.n_arms)
        
    def select_price(self):
        if self.n_iters < self.n_arms:
            a = np.zeros(self.n_arms)
            a[self.n_iters] = 1
            return a
        else:
            return self.ucb()
    def update(self, arm: int, reward: int):
        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1
        
    def ucb(self):
        ucb = self.arms_states / self.arms_actions
        ucb += np.sqrt(2 * np.log(self.n_iters) / (self.arms_actions+ 1e-5))
        return ucb
    
    
    
class ThompsonSamplingARPU:
    def __init__(self, price_points, prior_sigma=10000):
        self.price_points = price_points
        self.n = np.zeros(len(price_points))
        self.mu = np.zeros(len(price_points))
        self.sigma = np.zeros(len(price_points))
        self.post_sigma = np.sqrt((1 / prior_sigma ** 2 + self.n / self.sigma ** 2) ** -1) 
        self.values = [[] for x in range(len(price_points))]
        self.prior_sigma = prior_sigma
        self.iteration = 0
        
    def select_price(self):
        if self.iteration < len(self.price_points) :
            a = np.zeros(len(self.price_points))
            a[self.iteration] = 1
            return a
        samples = np.random.normal(self.mu, self.post_sigma)
        return samples

    def update(self, selected_price, df):
        self.iteration+=1
        self.n[selected_price] += len(df)
        self.values[selected_price].append(df)
        self.sigma[selected_price] = np.concatenate(self.values[selected_price]).std()

        if self.iteration > len(self.price_points):
            self.mu = [np.sum(np.concatenate(x)) for x in self.values]/ self.n
            self.post_sigma = np.sqrt((1 / self.prior_sigma ** 2 + self.n / self.sigma ** 2) ** -1)
            
class ThompsonSamplingBeta:

    def __init__(self, price_points):
        self.price_points = price_points
        self.alpha = np.ones(len(price_points))
        self.beta = np.ones(len(price_points))

    def select_price(self):
        samples = np.random.beta(self.alpha, self.beta)
        samples *= self.price_points
        return samples

    def update(self, arm, alpha,beta):
            self.alpha[arm] += alpha
            self.beta[arm] += beta


# In[5]:


class ThompsonSamplingGamma:

    def __init__(self, price_points):
        self.price_points = price_points
        self.alpha = np.ones(len(price_points))
        self.beta = np.ones(len(price_points))

    def select_price(self):
        samples = np.random.gamma(self.alpha, 1/self.beta)
        samples *= self.price_points
        return samples

    def update(self, arm, alpha,beta):
            self.alpha[arm] += alpha
            self.beta[arm] += beta


# In[6]:



from scipy.optimize import curve_fit

def linear_demand(x, a, b):
    return (-a*x) + b

def hyperbolic_demand(x, a, b):
    return (-a/x) + b

def quadratic_demand(x, a, b, c):
    return a - b * x + c * x ** 2


        
class UCB_QBC():
    def __init__(self, price_points):
        self.price_points = price_points
        self.n_arms = len(self.price_points)
        self.n_iters = 0
        self.arms_states = np.zeros(self.n_arms)
        self.arms_actions = np.zeros(self.n_arms)
        self.demands = {i: [] for i in range(self.n_arms)}
        
        self.params = {
            "linear_demand": None,       # a, b
            "quadratic_demand": None,  # a, b, c
        }
        
    def select_price(self):
        if self.n_iters < 3:
            a = np.zeros(self.n_arms)
            a[self.n_iters] = 1
            return a
        else:
            return self.ucb()
        
        
    def update(self, arm: int, reward: int, demand: int):
        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1
        self.demands[arm].append(demand)
        if self.n_iters >= 3:
            self.fit_models()    
    
    
    def ucb(self):
        lmbd = 1
        exploration = np.sqrt(2 * np.log(self.n_iters) / (self.arms_actions  + 1e-5))
        ucb_values = np.array([self.expected_reward(price)[0] * price for price in self.price_points])
        ucb = np.add(ucb_values,exploration , out=ucb_values, casting="unsafe")
        qbc = np.array([self.expected_reward(price)[1] for price in self.price_points])
        ucb_qbc = lmbd * ucb + (1 - lmbd) * qbc
        return ucb_qbc
    
    
       
    
    def fit_models(self):
        x_data = np.array(self.price_points)
        y_data = np.array([np.mean(self.demands[i]) if self.demands[i] else 0 for i in range(self.n_arms)])
       
        indexes = np.where(y_data !=0)
        x_data = x_data[indexes]
        y_data = y_data[indexes]
        self.params = {}
       
        for model, func in zip(["linear_demand", 'quadratic_demand'],
                               [linear_demand, quadratic_demand]):
            try:
                popt, _ = curve_fit(func, x_data, y_data)
                if popt is not None and not np.any(np.isnan(popt)):
                    self.params[model] = popt
                else:
                    self.params[model] = None
            except Exception as e:
                print(f"Couldn't fit {model} model. Reason: {e}")
                self.params[model] = None
                
                
    def expected_reward(self, x):
        rewards = []

        # Linear Demand Model
        if self.params['linear_demand'] is not None:
            a, b = self.params["linear_demand"]
            Q = linear_demand(x, a, b)
            rewards.append(Q)

        if self.params['quadratic_demand'] is not None:
            a, b, c= self.params["quadratic_demand"]
            Q = quadratic_demand(x, a, b, c)
            rewards.append(Q)

        if rewards:
            return  np.mean(rewards), np.sqrt(np.sum(np.var(rewards))) / np.shape(rewards)[0]  # average of the expected rewards from all valid models
        return 0,0


# In[7]:


# def fit_curve(func,x_data,y_data):
#     plt.plot(y_data, x_data, 'b-', label='data')
#     popt, pcov = curve_fit(func, x_data, y_data)
#     plt.plot(func(x_data, *popt),x_data, 'r-')
#     plt.xlabel("Спрос")
#     plt.ylabel("Цены")    


# def linear_demand(x, a, b):
#     return (-a*x) + b

# def hyperbolic_demand(x, a, b):
#     return (-a/x) + b

# def exponential_demand(x, a, b, c):
#     return -np.exp((a*x) + b) + c

# def power_demand(x, a, b, c):
#     return (a*x)**b + c 

# def exponential_2(x, a, b):
#     return (a * np.exp(b * x)) 

# def quadratic_demand(x, a, b, c):
#     return a - b * x + c * x ** 2

# fit_curve(hyperbolic_demand, x_data,y_data)
# fit_curve(quadratic_demand, x_data,y_data)


# In[8]:


class BanditRunner:
    def __init__(self, bandit, name, history):
        self.bandit = bandit
        self.name = name
        self.history = history

    def run(self, df1, df2):
        samples_1 = self.bandit[0].select_price()
        samples_2 = self.bandit[1].select_price()
        index = np.argmax(samples_1 + samples_2)
        price_1 = self.bandit[0].price_points[index]
        price_2 = self.bandit[1].price_points[index]
        
        
        reward_1 = int(df1[df1['price'] == price_1]['revenue'].sum())
        demand_1 = int(df1[df1['price'] == price_1]['demand'].sum())
        
        reward_2 = int(df2[df2['price'] == price_2]['revenue'].sum())
        demand_2 = int(df2[df2['price'] == price_2]['demand'].sum())
        
        if self.name in ['eps','ucb']:
            self.bandit[0].update(index, reward_1)
            self.bandit[1].update(index, reward_2)
        elif self.name in ['ucb_qbc']:
            self.bandit[0].update(index, reward_1, demand_1)
            self.bandit[1].update(index, reward_2, demand_2)
        elif self.name in ['ts_arpu']:
            self.bandit[0].update(index, df1[df1['price'] == price_1]['revenue'])
            self.bandit[1].update(index, df2[df2['price'] == price_2]['revenue'])
        elif self.name in ['ts_beta','ts_gamma']:
            beta = df1[(df1['price'] == price_1) & (df1['demand'] == 0)].shape[0]
            self.bandit[0].update(index, demand_1,beta)
            
            beta = df2[(df2['price'] == price_2) & (df2['demand'] == 0)].shape[0]
            self.bandit[1].update(index, demand_2,beta)
            
        self.history.append(reward_1 + reward_2)
        

import matplotlib.pyplot as plt
def plot_results(history_epsilon,history_ucb, history_qbc, history_ts, history_ts_beta,history_ts_gamma, price_points):
    plt.figure(figsize=(20,10))
    plt.plot(history_epsilon, label='Epsilon Greedy')
    plt.plot(history_ucb, label='UCB')
    plt.plot(history_qbc, label='UCB+QBC')
    plt.plot(history_ts, label='ts_ARPU')
    plt.plot(history_ts_beta, label='ts_beta')
    plt.plot(history_ts_gamma, label='ts_gamma')
    plt.legend()
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Reward')
    plt.show()




# In[9]:


np.random.seed(42)
num_prices = 5
base_price = 200
max_price = 2001
step_size = int((max_price - base_price) / (num_prices - 1))
price_points = np.arange(base_price, 2001, step_size)

#price_points = np.linspace(200,1000,100)
# Usage:
simulator = DemandSimulator(user_count=10000,demand_exponent=-0.0004, max_demand=10, min_demand=0, high_demand_users_ratio=0.1)

df_1 = simulator.simulate_user_behaviour_1(price_points)
df_2 = simulator.simulate_user_behaviour_2(price_points)
df_1['index'] = range(df_1.shape[0])
df_2['index'] = range(df_2.shape[0])
df = pd.merge(df_1, df_2, on = 'index', suffixes = ('_1', '_2'))
simulator.plot_demand_and_revenue(df_1)
simulator.plot_demand_and_revenue(df_2)
simulator.plot_demand_products(df)


# In[16]:



NUMBER_OF_EPOCH = 30
history_epsilon, history_ucb, history_qbc, history_ts, history_ts_beta, history_ts_gamma = [], [], [], [], [], []
bandits = [
    BanditRunner([EpsGreedy(price_points, 0.1), EpsGreedy(price_points, 0.1)], 'eps', history_epsilon),
    BanditRunner([UCB(price_points), UCB(price_points)], 'ucb',history_ucb),
    BanditRunner([UCB_QBC(price_points), UCB_QBC(price_points)], 'ucb_qbc' ,history_qbc),
    BanditRunner([ThompsonSamplingARPU(price_points), ThompsonSamplingARPU(price_points)], 'ts_arpu', history_ts),
    BanditRunner([ThompsonSamplingBeta(price_points),ThompsonSamplingBeta(price_points)], 'ts_beta', history_ts_beta),
    BanditRunner([ThompsonSamplingGamma(price_points),ThompsonSamplingGamma(price_points)], 'ts_gamma', history_ts_gamma)
    
]
weekdays = [x for x in range(0, NUMBER_OF_EPOCH + 1, 7)]
weekdays.extend(([x+1 for x in range(0, NUMBER_OF_EPOCH + 1, 7)]))

for i in range( NUMBER_OF_EPOCH+1):
#     if i%50 == 0:
#         demand_exponent += 0.0001
#         simulator = DemandSimulator(user_count=user_count,demand_exponent=demand_exponent, max_demand=10, min_demand=0, high_demand_users_ratio=0.2)
#         simulator.plot_demand_and_revenue(simulator.simulate_user_behaviour(price_points))        
    
#     if i in weekdays:
#         user_count = 1000
#     else:
#         user_count = 10_000
    df_1 = simulator.simulate_user_behaviour_1(price_points)
    df_2 = simulator.simulate_user_behaviour_2(price_points)
    for bandit_runner in bandits:
        bandit_runner.run(df_1,df_2)


# In[17]:




history_epsilon = np.cumsum(np.array(history_epsilon).astype(float))
history_ucb = np.cumsum(np.array(history_ucb).astype(float))
history_qbc = np.cumsum(np.array(history_qbc).astype(float))
history_ts = np.cumsum(np.array(history_ts).astype(float))
history_ts_beta = np.cumsum(np.array(history_ts_beta).astype(float))
history_ts_gamma = np.cumsum(np.array(history_ts_gamma).astype(float))


# In[18]:


print(price_points[np.argmax(bandits[0].bandit[0].select_price() + bandits[0].bandit[1].select_price())],bandits[0].bandit[0])
print(price_points[np.argmax(bandits[1].bandit[0].select_price() + bandits[1].bandit[1].select_price())],bandits[1].bandit[0])
print(price_points[np.argmax(bandits[2].bandit[0].select_price() + bandits[2].bandit[1].select_price())],bandits[2].bandit[0])
print(price_points[np.argmax(bandits[3].bandit[0].select_price() + bandits[3].bandit[1].select_price())],bandits[3].bandit[0])
print(price_points[np.argmax(bandits[4].bandit[0].select_price() + bandits[4].bandit[1].select_price())],bandits[4].bandit[0])
print(price_points[np.argmax(bandits[5].bandit[0].select_price() + bandits[5].bandit[1].select_price())],bandits[5].bandit[0])


# In[19]:


history_epsilon[-1], history_ucb[-1], history_qbc[-1], history_ts[-1],history_ts_beta[-1],history_ts_gamma[-1]


# In[20]:


plot_results(history_epsilon, history_ucb, history_qbc, history_ts,history_ts_beta, history_ts_gamma, price_points)

