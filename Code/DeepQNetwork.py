import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


np.random.seed(1)
torch.manual_seed(1)


# Generate a Q network
# This Q network uses 2 fully connected layers and 1 final linear fully connected layer
class QNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_space, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, action_space)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.layer3)
        return model(x)


# Create an agent that uses the Q Network defined above
class DQNAgent:
    def __init__(self, state_space, action_space, args):
        self.policy_net = QNetwork(state_space, action_space)
        self.epsilon = args["epsilon"]
        self.learning_rate = args["learning_rate"]
        self.step_size = args["step_size"]
        self.discount = args["gamma"]
        self.decay_rate = args["decay_rate"]
        self.objective = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.step_size, self.discount)

    # Choose an epsilon-greedy action
    def choose_action(self, env, Q):
        if np.random.random() <= 1 - self.epsilon:
            # Choose the currently best known action
            action = torch.argmax(Q).item()
        else:
            # Try a random action
            action = env.action_space.sample()
        return action

    # Get the Q value of the state
    def get_Q(self, state):
        state_tensor = torch.tensor(state, requires_grad=False).float()
        Q = self.policy_net(state_tensor)
        return Q

    # Update the neural network using a target Q value
    def update_network(self, Q, next_state, choice, reward):
        # Forward pass using next state
        next_state_tensor = torch.tensor(next_state, requires_grad=False).float()
        maxQ1 = torch.max(self.policy_net(next_state_tensor))
        # Create a target Q value for training the policy
        Q_target = Variable(Q.clone())
        # Calculate the loss between Q
        Q_target[choice] = reward + self.discount*maxQ1.detach()
        loss = self.objective(Q, Q_target)
        # Update the Q network (backward pass)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), reward

    # Update the epsilon for epsilon-greedy policy
    def update_epsilon(self):
        self.epsilon *= self.decay_rate

    def update_scheduler(self):
        self.scheduler.step()
