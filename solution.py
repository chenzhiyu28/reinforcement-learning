# Some parts of this code were adapted from COMP3702 Tutorial 9
# solution code ("GridWorld_RL_soln(1).py" by njc, available on COMP3702
# Blackboard page, retrieved 13 October 2022)


import sys
import time
from pprint import pprint

from constants import *
from environment import *
from state import State
import random
import numpy as np



"""
solution.py

This file is a template you should use to implement your solution.

You should implement code for each of the TODO sections below.

COMP3702 2022 Assignment 3 Support Code

Last updated by njc 12/10/22
"""


class RLAgent:

    #
    # TODO: (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment
        self.learning_rate = environment.alpha
        self.state = environment.get_init_state()
        self.q_EPSILON = 0.8

        self.q_table = dict()
        self.states = []
        self.get_all_states()

        self.s_EPSILON = 0.8
        self.q_table_SARSA = self.q_table
        self.sarsa_next_action = FORWARD

        self.rewards = []

    # all possible states
    def get_all_states(self) -> None:
        for x in range(self.environment.n_rows):
            for y in range(self.environment.n_cols):
                # an available position for agent
                if self.environment.obstacle_map[x][y] == \
                        self.environment.hazard_map[x][y]:
                    for orientation in ROBOT_ORIENTATIONS:
                        self.states.append(State(self.environment, (x, y),
                                                 orientation))
                        for action in ROBOT_ACTIONS:
                            self.q_table[State(self.environment, (x, y),
                                               orientation), action] = -1

    # === Q-learning ===================================================================================================
    def q_next_iteration(self):
        """
        Write a method to update your agent's q_values here
        Include steps to generate new state-action q_values as you go
        """

        # Choose an action, simulate it, and receive a reward
        # 1.选择action 2.更新 执行action之后 的新状态

        state = self.state

        if random.random() > self.q_EPSILON:
            action = self.q_learn_select_action(state)
        else:
            action = random.choice(ROBOT_ACTIONS)

        reward, next_state = self.environment.perform_action(state, action)

        self.rewards.append(reward)

        # Update q-value for the (state, action) pair
        # 1.获得q table中已有的 q(state,action) 2.获得 执行action之后, new_state的best_action(exploit)
        old_q = self.q_table.get((state, action), 0)
        best_next_action = self.q_learn_select_action(next_state)

        # 获取new_state 的 q(state,action)
        best_next_q = self.q_table.get((next_state, best_next_action), 0)

        # 如果完成了,重置到起点
        if next_state.robot_posit in self.environment.target_list:
            next_state = self.environment.get_init_state()
            #next_state = random.choice(self.states)
            # 1: 10/9.3/10
            best_next_q = 1000

            self.rewards.append("stop")

            self.q_EPSILON *= 0.999

        # 套公式
        self.best_next_q = best_next_q

        target = reward + self.environment.gamma * best_next_q
        new_q = old_q + self.learning_rate * (target - old_q)

        # print(old_q-new_q)
        # 更新 q table
        self.q_table[(state, action)] = new_q

        self.state = next_state

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        i = 0
        start = time.time()

        while (time.time() - start) < self.environment.training_time_tgt:
            i += 1
            self.q_next_iteration()

            if self.environment.get_total_reward() < 1.05 * self.environment.training_reward_tgt:
                break

        # diagram

        record = []
        sum = 0

        for i in self.rewards:
            # if not solved
            if type(i) != str:
                sum += i

            else:
                record.append(round(sum, 1))
                sum = 0

        """
        print(len(record))
        pprint(record[-50:])
        """

        sum2 = 0
        number = 0
        ave = []
        for i in record:
            sum2 += i
            number += 1
            if number == 50:
                ave.append(round(sum2/number, 1))
                number = 0
                sum2 = 0

        print(len(ave))
        pprint(ave[-10:])
        print("------------")
        y_axis = "average reward"
        x_axis = "t"

        x = np.array(list(range(1, len(ave)+1)))
        y = np.array(ave)

        return x, y

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        best_a = FORWARD
        for action in ROBOT_ACTIONS:
            if self.q_table.get((state, action), -0.1) > self.q_table.get(
                    (state, best_a), -0.1):
                best_a = action
        return best_a

    # === SARSA ========================================================================================================

    def s_next_iteration(self):
        """
        Write a method to update your agent's q_values here
        Include steps to generate new state-action q_values as you go
        """

        # Choose an action, simulate it, and receive a reward
        # 1.选择action 2.更新 执行action之后 的新状态
        action = self.sarsa_next_action

        state = self.state

        reward, next_state = self.environment.perform_action(state, action)
        self.rewards.append(reward)

        # Update q-value for the (state, action) pair
        # 1.获得q table中已有的 q(state,action) 2.获得 执行action之后, new_state的best_action(exploit)
        old_q = self.q_table_SARSA.get((state, action), 0)
        next_action = self.sarsa_select_action(next_state)
        self.sarsa_next_action = next_action

        # 获取new_state 的 q(state,action)
        next_q = self.q_table_SARSA.get((next_state, next_action), 0)

        # 如果完成了,重置到起点
        if next_state.robot_posit in self.environment.target_list:
            #next_state = self.environment.get_init_state()
            """
            next_state = random.choice(self.states)
            next_q = 388

            self.q_EPSILON *= 0.999
            """
            next_state = self.environment.get_init_state()
            next_q = 1000
            self.rewards.append("stop")
            self.q_EPSILON *= 0.999

        # 套公式
        target = reward + self.environment.gamma * next_q
        new_q = old_q + self.learning_rate * (target - old_q)

        # print(old_q-new_q)
        # 更新 q table
        self.q_table_SARSA[(state, action)] = new_q

        self.state = next_state

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        i = 0
        start = time.time()

        while (time.time() - start) < self.environment.training_time_tgt:
            i += 1
            self.s_next_iteration()

            if self.environment.get_total_reward() < 1.05 * self.environment.training_reward_tgt:
                break

        # diagram

        record = []
        sum = 0

        for i in self.rewards:
            # if not solved
            if type(i) != str:
                sum += i

            else:
                record.append(round(sum, 1))
                sum = 0

        """
        print(len(record))
        pprint(record[-50:])
        """

        sum2 = 0
        number = 0
        ave = []
        for i in record:
            sum2 += i
            number += 1
            if number == 50:
                ave.append(round(sum2/number, 1))
                number = 0
                sum2 = 0

        print(len(ave))
        pprint(ave[-10:])
        print("------------")
        y_axis = "average reward"
        x_axis = "t"

        x = np.array(list(range(1, len(ave)+1)))
        y = np.array(ave)

        return x, y


    # 包含随机事件的 select
    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """

        if random.random() > self.s_EPSILON:
            best_a = random.choice(ROBOT_ACTIONS)
        else:
            best_a = FORWARD
            for action in ROBOT_ACTIONS:
                if self.q_table.get((state, action), -0.1) > self.q_table.get(
                        (state, best_a), -0.1):
                    best_a = action

        return best_a

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    #

