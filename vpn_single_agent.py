import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt


class EncoderModel(nn.Module):
    """ Encodes the current state into a representation. """

    def __init__(self, input_channels=3, hidden_dims=64):
        super(EncoderModel, self).__init__()

        # Define convolutional layers according to the specification
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # Padding added for alignment
        self.conv4 = nn.Conv2d(128, hidden_dims, kernel_size=1)  # 1x1 convolution to maintain output shape
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Apply the convolutional layers with ReLU activations
        # print('x ins ' + str(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Apply the 1x1 convolution to reduce the spatial dimensions to (1, 1)
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # print('encoode x shape  ' + str(x.shape))
        return x


class OptionConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, num_options):
        super(OptionConv, self).__init__()
        # Create different convolution layers for each option
        self.option_convs = nn.ModuleList([
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)
            for _ in range(num_options)
        ])

    def forward(self, x, option_idx):
        """
        x: input tensor
        option_idx: the index of the selected option (based on some criteria)
        """
        # Apply the convolution based on the selected option
        conv_layer = self.option_convs[option_idx]  # Select the appropriate convolution layer
        return conv_layer(x)


class TransitionModel(nn.Module):
    """ Predicts the next state representation given the current representation and an action. """

    def __init__(self, input_channels=64, num_options=3):
        super(TransitionModel, self).__init__()

        # Define the OptionConv Layer (with 3 options)
        self.option_conv1 = OptionConv(input_channels, 64, kernel_size=3, stride=1, padding=1,
                                       num_options=num_options)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x, option_idx):
        """
        x: current abstract state (input tensor)
        residual: previous abstract state (residual connection)
        option_idx: the selected option index
        """
        # Apply the first OptionConv layer based on the selected option

        residual = x
        # print('residual shape ' + str(residual.shape))
        x = F.relu(self.option_conv1(x, option_idx))

        # Apply further convolutions
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Add the residual connection (difference between previous and current state)
        x += residual
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x


class OutComeModel(nn.Module):
    """ Predicts the reward for a given state representation and action. """

    def __init__(self, input_channels=64, num_options=3):
        super(OutComeModel, self).__init__()
        # First convolutional layer (OptionConv)

        self.option_conv1 = OptionConv(input_channels, 64, kernel_size=3, stride=1, padding=1,
                                       num_options=num_actions)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Another 64 filters of size 3x3
        self.fc1 = nn.Linear(64, 32)  # Fully connected layer (64 hidden units)
        self.fc2 = nn.Linear(32, 2)  # Output layer with 2 units

    def forward(self, x, option):
        x = F.relu(self.option_conv1(x, option))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        # print('flatten shape ' + str(x.shape))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ValueModel(nn.Module):
    """ Predicts the value for a given state representation. """

    def __init__(self, abs_state_dim):
        super(ValueModel, self).__init__()
        self.fc = nn.Linear(abs_state_dim, 1)

    def forward(self, abs_state):
        abs_state = torch.flatten(abs_state)
        value = self.fc(abs_state)
        return value


class ValuePredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(ValuePredictionNetwork, self).__init__()
        self.encode_model = EncoderModel(input_dim, hidden_dim)
        self.transition_model = TransitionModel(hidden_dim, num_actions)
        self.outcome_model = OutComeModel(hidden_dim, num_actions)
        self.value_model = ValueModel(hidden_dim)
        self.num_actions = num_actions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, state, action):
        """
        Forward pass through the VPN with planning.

        Args:
            state (torch.Tensor): Initial state.
            action_sequence (torch.Tensor): Sequence of actions for planning.

        Returns:
            value (torch.Tensor): Predicted value of the state-action sequence.
            rewards (list): Predicted rewards for each transition in the action sequence.
        """
        # Encode the current state
        # state_rep = self.encode_model(state)

        # Initialize lists for predicted rewards and values
        rewards = []
        # print('state inside vpn ' + str(state.shape))
        action = action.to(self.device)

        action_index = torch.argmax(action).item()
        if state.size() == torch.Size([1, 3, 96, 96]):
            abs_state = self.encode_model(state).to(self.device)
        else:
            abs_state = self.transition_model(state, action_index)

        # print('abs state sgao ' + str(abs_state.shape))

        # TODO need to check if transition state or state is input to value function
        value = self.value_model(abs_state)
        outcome = self.outcome_model(abs_state, action_index)
        # print('outcome ' + str(outcome))
        reward = outcome[1]
        discount = outcome[0]

        return reward, discount, value, abs_state


def q_plan(state, option, depth, vpn, env, b=2):
    # Forward pass to get the initial reward, discount, value, and next transition state
    reward, discount, value, transition_state = vpn.forward(state, option)

    # Initialize lists to store reward, discount, and value at each depth
    path_rewards = [reward]
    path_discounts = [discount]
    path_values = [value]
    best_path_actions = [option]
    # Base case for recursion
    if depth == 1:
        # Return Q-value with only the immediate reward and discount applied
        return reward + discount * value, path_rewards, path_discounts, path_values, best_path_actions
    else:
        # To hold the Q-values and selected paths from each option
        q1_values = []

        # Generate Q1 values for all actions in the action space
        for action in range(env.action_space.n):
            action_ohe = torch.zeros(env.action_space.n)
            action_ohe[action] = 1
            r, d, v, s_1 = vpn.forward(transition_state, action_ohe)
            q_value = r + d * v
            q1_values.append(q_value)

        # Select top-b best actions based on Q1 values
        q1_values = torch.stack(q1_values, dim=0)
        _, indices = torch.topk(q1_values, b, dim=0, largest=True, sorted=False)
        A = [i.item() for i in indices]

        # To keep track of the best Q-value and the path chosen to achieve it
        best_q_value = -float('inf')
        best_rewards, best_discounts, best_values = [], [], []

        # Recursively calculate Q-values for top options in A
        for option in A:

            option_ohe = torch.zeros(env.action_space.n)
            option_ohe[option] = 1
            q_val, rewards, discounts, values, reward_path_actions = q_plan(transition_state, option_ohe, depth - 1,
                                                                            vpn,
                                                                            env, b)

            # Calculate the cumulative Q-value
            cumulative_q_value = reward + (1 / depth) * value + ((depth - 1) / depth) * q_val

            # Update if we found a better path
            if cumulative_q_value > best_q_value:
                best_q_value = cumulative_q_value
                best_rewards = [reward] + rewards
                best_discounts = [discount] + discounts
                best_values = [value] + values
                best_reward_path_actions = [option] + reward_path_actions

    # Return the best Q-value found along with the path details
    return best_q_value, best_rewards, best_discounts, best_values, best_reward_path_actions


def calculate_loss(target_val, target_reward, target_discount, pred_val, pred_reward, pred_discount):
    # n_step_loss = 0.0
    # for t in range(len(target_val), -1, -1):
    #     for k in range(len(pred_val[t])):
    #
    #         n_step_loss += (target_val[t] - pred_val[t][k])**2 + (target_reward[t] - pred_reward[t][k])**2 + \
    #                        (target_discount[t] - pred_discount[t][k])**2
    #
    # return n_step_loss

    loss_val = (target_val - torch.tensor(pred_val).cuda()) ** 2  # Unsqueeze to align dimensions for broadcasting
    loss_reward = (target_reward - torch.tensor(pred_reward).cuda()) ** 2
    loss_discount = (target_discount - torch.tensor(pred_discount).cuda()) ** 2

    # Sum the losses over the batch and n_steps
    n_step_loss = torch.sum(loss_val + loss_reward + loss_discount)
    # print('loss ' + str(n_step_loss))
    return n_step_loss


def epsilon_greedy_policy(vpn, env, state, depth, eps, b):
    # print('state ' + str(state))
    with torch.no_grad():
        q_values = []
        for action in range(env.action_space.n):
            # print('epsilon greedy action ' + str(action))
            action_ohe = torch.zeros(env.action_space.n)
            action_ohe[action] = 1
            q_value, _, _, _, _ = q_plan(state, action_ohe, depth, vpn, env, b)
            q_values.append(q_value)

    if random.random() <= eps:
        selected_action = np.random.randint(0, env.action_space.n)
        selected_action_ohe = torch.zeros(env.action_space.n)
        selected_action_ohe[action] = 1
        # print('random action ' + str(selected_action))
    else:
        selected_action = q_values.index(max(q_values))
        selected_action_ohe = torch.zeros(env.action_space.n)
        selected_action_ohe[action] = 1
        # print('action selected ' + str(selected_action))

    return selected_action_ohe, selected_action


# def state_to_vector(state, dims):

def update_parameters(train_vpn, target_vpn):
    target_vpn.load_state_dict(train_vpn.state_dict())


# Example usage
if __name__ == "__main__":

    env = gym.make("CarRacing-v2", continuous=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_episodes = 3000
    # print('observation space ' + str(env.observation_space.shape))
    input_dim = env.observation_space.shape[-1]
    # print('action space ' + str(env.action_space))
    hidden_dim = 64  # Hidden dimension for the state representation
    num_actions = env.action_space.n
    # print('number of actions ' + str(num_actions))
    action_space = [torch.tensor([1, 0], dtype=torch.float),
                    torch.tensor([0, 1], dtype=torch.float)]  # Example action space

    depth = 3  # Planning depth
    k = 3
    b = 2  # Top 'b' actions to consider
    lr = 0.0001
    n = 10
    # Initialize the VPN (Value Prediction Network)
    vpn = ValuePredictionNetwork(input_dim, hidden_dim, num_actions).to(device)
    vpn_target = ValuePredictionNetwork(input_dim, hidden_dim, num_actions).to(device)

    optimizer = torch.optim.Adam(vpn.parameters(), lr=lr)
    eps = 0.99
    eps_decay = 0.9998
    eps_min = 0.001

    for episode in range(num_episodes):

        T = 0
        state, _ = env.reset()
        is_terminal = False

        local_episode = 0
        total_reward = 0
        while not is_terminal:

            local_episode += 1
            # print('state shape ' + str(state[0].shape))
            rewards = []
            gammas = []
            states = []
            actions = []
            states = []
            for t in range(n):
                # print('step t= ' + str(t))
                # print('state shape ' + str(state.shape))
                if t == 0:
                    states.append(state)
                state = np.transpose(state, (2, 0, 1))  # Convert (96, 96, 3) to (3, 96, 96)
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(
                    0).cuda()  # Add batch dimension and move to GPU
                # state.permute(2, 0, 1)
                action_ohe, action = epsilon_greedy_policy(vpn, env, state, depth, eps, b)
                _, gamma, value, _ = vpn.forward(state, action_ohe)
                # print('taking action ' + str(action))
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                rewards.append(reward)
                gammas.append(gamma)
                states.append(state)
                actions.append(action)
                if terminated or truncated:
                    if terminated:
                        print('terminal reached')
                    elif truncated:
                        print('truncated')
                    is_terminal = True
                    break
            # print('num states ' + str(len(states)))
            # print('num rewards ' + str(len(rewards)))
            if is_terminal:
                R = 0
            else:
                last_state = states[-1]
                last_state = np.transpose(last_state, (2, 0, 1))  # Convert (96, 96, 3) to (3, 96, 96)
                last_state = torch.tensor(last_state, dtype=torch.float32).unsqueeze(
                    0).cuda()  # Add batch dimension and move to GPU
                q_depth = []
                with torch.no_grad():
                    q_values = []
                    for action in range(env.action_space.n):
                        action_ohe = torch.zeros(env.action_space.n)
                        action_ohe[action] = 1
                        q_value, _, _, _, _ = q_plan(last_state, action_ohe, depth, vpn, env, b)
                        q_depth.append(q_value)

                R = max(q_depth)
            loss = 0.0
            for t in range(len(rewards) - 1, -1, -1):
                # print('t ' + str(t))
                R = rewards[t] + gammas[t] * R
                action = actions[t]
                action_ohe = torch.zeros(env.action_space.n)
                action_ohe[action] = 1

                state_t = np.transpose(states[t], (2, 0, 1))  # Convert (96, 96, 3) to (3, 96, 96)
                state_t = torch.tensor(state_t, dtype=torch.float32).unsqueeze(
                    0).cuda()  # Add batch dimension and move to GPU

                q_value, best_rewards, best_discounts, best_values, best_path = q_plan(state_t, action_ohe, depth,
                                                                                       vpn_target,
                                                                                       env,
                                                                                       b)

                loss += calculate_loss(R, rewards[t], gammas[t], best_values, best_rewards, best_discounts)
            print('global episode ' + str(episode) + ' local episode ' + str(local_episode) + ' loss ' + str(
                loss) + ' total reward ' + str(total_reward))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (episode * local_episode) % 10000 == 0:
                update_parameters(vpn, vpn_target)

            eps = max(eps * eps_decay, eps_min)
