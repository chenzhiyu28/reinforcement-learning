from environment import *
from state import *
from constants import *
from solution import *
import random
import pprint
import numpy as np
import matplotlib.pyplot as plt

"""
env = Environment("ex3.txt")
# (0,2)
# q table的值自动消失了
# 公式无效，自动覆盖值???

q1 = RLAgent(Environment("ex3.txt"))
q1.learning_rate = 0.01
x1, y1 = q1.q_learn_train()

q2 = RLAgent(Environment("ex3.txt"))
q2.learning_rate = 0.1
x2, y2 = q2.q_learn_train()

q3 = RLAgent(Environment("ex3.txt"))
q3.learning_rate = 0.8
x3, y3 = q3.q_learn_train()

plt.plot(x1, y1, label = "α = 0.01")
plt.plot(x2, y2, label = "α = 0.1")
plt.plot(x3, y3, label = "α = 0.8")
plt.legend()
plt.show()
"""


q1 = RLAgent(Environment("ex3.txt"))
x1, y1 = q1.q_learn_train()


s1 = RLAgent(Environment("ex4.txt"))
x2, y2 = s1.sarsa_train()

plt.plot(x1, y1, label = "Q-learning")
plt.plot(x2, y2, label = "SARSA")
plt.legend()
plt.show()