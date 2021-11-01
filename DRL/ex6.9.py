import numpy as np
import matplotlib.pyplot as plt
from gridworld_random import WindyGridworld

class Agent:
	def __init__(self, alpha=0.5, epsilon=0.1, action_number=4):
		self.learning_rate = alpha
		self.epsilon = epsilon
		self.action_num = action_number
		self.q_value = np.zeros((10+1, 7+1, action_number))
		self.actions = self.action_map()

	def policy(self, state):
		x, y = state
		if np.random.uniform() <= self.epsilon:
			action = np.random.randint(0, self.action_num)
			return self.actions[action]
		else:
			action = np.argmax(self.q_value[x][y])
			return self.actions[action]


	def learn(self, s, a, r, s_, a_):
		a_ = self.actions.tolist().index(a_)
		a = self.actions.tolist().index(a)
		td_err = r + self.q_value[s_[0]][s_[1]][a_] - self.q_value[s[0]][s[1]][a]
		self.q_value[s[0]][s[1]][a] = self.q_value[s[0]][s[1]][a] + \
		                                                 self.learning_rate * td_err


	def action_map(self):
		if self.action_num == 4:
			return np.array(["left", "up", "right", "down"])
		elif self.action_num == 8:
			return np.array(["left", "left-up", "up", "right-up", "right",    \
			                                          "right-down", "down", "left-down"])
		else:
			return np.array(["left", "left-up", "up", "right-up", "right",     \
			                                   "right-down", "down", "left-down", "stay"])


def rl():
	# world = WindyGridworld(is_eight_action=True, ninth_action=True)
	world = WindyGridworld()
	agent = Agent()

	start_state = (1, 4) #(0, 3)
	end_state = (8, 4)

	episode = 170
	stepPerEpside = []
	step = 0
	while episode > 0:
		state = start_state
		while True:
			action = agent.policy(state)
			next_state, reward = world.step(state, action)
			# if episode == 169:
			# 	print(state, action)

			if next_state == end_state:
				break
			next_action = agent.policy(next_state)
			agent.learn(state, action, reward, next_state, next_action)
			state = next_state
			step = step + 1
		episode = episode - 1
		stepPerEpside.append(step)
		print(stepPerEpside[0])
		# print("episode: {} step: {}\n".format(episode, step))
	# print(len(stepPerEpside))\
	print(agent.q_value[1][4])

	plt.figure()
	plt.plot(stepPerEpside, range(170))
	plt.show()

if __name__ == "__main__":
	rl()




















