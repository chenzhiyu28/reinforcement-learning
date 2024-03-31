import sys
import time
from constants import *
from environment import *
from state import State
import random

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
        self.q_EPSILON = 0.2
        self.q_current_reward = 0

        self.q_table = dict()
        self.get_all_states()

    # all possible states
    def get_all_states(self) -> None:
        for x in range(self.environment.n_rows):
            for y in range(self.environment.n_cols):
                for action in ROBOT_ACTIONS:
                    # an available position for agent
                    if self.environment.obstacle_map[x][y] == self.environment.hazard_map[x][y]:
                        for orientation in ROBOT_ORIENTATIONS:
                            self.q_table[State(self.environment, (x, y), orientation), action] = -0.01
        print('执行')
    # === Q-learning ===================================================================================================

    # update q_table
    def q_next_iteration(self) -> None:
        # Choose an action, simulate it, and receive a reward
        # 1.选择action 2.更新 执行action之后 的新状态
        state = self.state

        if random.random() > self.q_EPSILON:
            action = self.q_learn_select_action(state)
        else:
            action = random.choice(ROBOT_ACTIONS)
        # print(action)

        reward, next_state = self.environment.perform_action(state, action)
        self.state = next_state

        # 如果到了 target, 重置agent position到初始
        if self.environment.is_solved(self.state):
            self.state = self.environment.get_init_state()

        # Update q-value for the (state, action) pair
        # 1.获得q table中已有的 q(state,action) 2.获得 执行action之后, new_state的best_action(exploit)
        old_q = self.q_table.get((state, action), 0)
        best_next_action = self.q_learn_select_action(next_state)

        # 获取new_state 的 q(state,action)
        best_next_q = self.q_table.get((next_state, best_next_action), 0)
        if next_state.robot_posit in self.environment.target_list:
            best_next_q = 0

        # 套公式
        target = reward + self.environment.gamma * best_next_q
        new_q = old_q + self.learning_rate * (target - old_q)
        # 更新 q table
        self.q_table[(state, action)] = new_q

        if state.robot_posit == (0, 2):
            print()
            print(state.robot_posit)
            print("action:", action)
            print("value: ", self.q_table[(state, action)])
            for action in ROBOT_ACTIONS:
                print(action, ":" ,self.q_table.get((state, action)))
            print("--------------")


    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        i = 0

        start = time.time()
        while (time.time() - start) < self.environment.training_time_tgt:
        #while ((time.time() - start) < self.environment.training_time_tgt) and self.environment.evaluation_reward_tgt:
            i += 1
            self.q_next_iteration()
            #print(self.environment.get_total_reward())

            if i > 45:
                break

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        best_q = float('-inf')
        best_a = FORWARD
        for action in ROBOT_ACTIONS:
            this_q = self.q_table.get((state, action))
            if this_q is not None and this_q > best_q:
                best_q = this_q
                best_a = action
        return best_a


    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # TODO: Implement your SARSA training loop here.
        #
        pass

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: (optional) Add any additional methods here.
    #
    #

