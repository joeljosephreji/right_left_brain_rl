import torch
import torch.nn as nn
import numpy as np
from models.policy import FixedNormal
from models.gating_network import GatingNetwork, StepGatingNetwork, EncoderGatingNetwork

from gatingcombine import GatingCombine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):

    def __init__(self, policy, encoder):
        super().__init__()
        self.policy = policy
        self.encoder = encoder

    def get_actor_params(self):
        return self.policy.get_actor_params()

    def get_critic_params(self):
        return self.policy.get_critic_params()

    def forward_actor(self, inputs):
        return self.policy.forward_actor(inputs)

    def forward_critic(self, inputs):
        return self.policy.forward_critic(inputs)

    def act(self, state, latent, belief=None, task=None, deterministic=False):
        return self.policy.act(state, latent, belief, task, deterministic)

    def get_value(self, state, latent, belief=None, task=None):
        value, _ = self.policy.forward(state, latent, belief, task)
        return value

    def evaluate_actions(self, state, latent, belief, task, action):
        """Call policy eval, set task, belief to None"""
        return self.policy.evaluate_actions(state, latent, belief, task, action)


class BiHemActorCritic(nn.Module):

    def __init__(
            self,
            left_policy,
            left_encoder,
            right_policy,
            right_encoder,
            dim_state,
            dim_action,
            init_std,
            gating_combination_method,
            use_action_in_gate=False,
            use_state_in_gate=False,
            use_gating_schedule=False,
            gating_schedule_type=None,
            gating_schedule_update=None,
            min_right_value=None,
            init_right_value=None
    ):
        super().__init__()
        self.left_actor_critic = ActorCritic(left_policy, left_encoder)
        self.right_actor_critic = ActorCritic(right_policy, right_encoder)

        if use_gating_schedule:
            self.gating_network = StepGatingNetwork(
                gating_schedule_type=gating_schedule_type,
                gating_schedule_update=gating_schedule_update,
                min_right_value=min_right_value,
                init_right_value=init_right_value
            )
        else:

            self.gating_network = EncoderGatingNetwork(
                take_action=use_action_in_gate,
                take_state=use_state_in_gate,
                dim_action=dim_action,
                dim_state=dim_state
            )
            # self.gating_network = GatingNetwork(
            #     dim_state + left_encoder.latent_dim * 2 + right_encoder.latent_dim * 2
            # )

        self.logstd = nn.Parameter(np.log(torch.zeros(dim_action) + init_std))
        self.min_std = torch.tensor([1.0e-6]).to(device)
        # self.max_std = torch.tensor([1.0e6]).to(device)
        # self.std = torch.tensor([init_std]).to(device)

        self.gating_combination_method = gating_combination_method

    def encoder(self, action, state, reward, value_errors, gate_values, hidden_state, return_prior=False, sample=False, detach_every=None):
        if isinstance(hidden_state, tuple):
            gate_hidden_state = hidden_state[0]
            left_hidden_state = hidden_state[1]
            right_hidden_state = hidden_state[2]

        else:
            raise ValueError

        _, left_latent_mean, left_latent_logvar, left_hidden_state = self.left_actor_critic.encoder(
            action,
            state,
            reward,
            left_hidden_state,
            return_prior=return_prior,
            sample=sample,
            detach_every=detach_every
        )
        # remove observable goals from right
        # state[...,-4] = 0
        _, right_latent_mean, right_latent_logvar, right_hidden_state = self.right_actor_critic.encoder(
            action,
            state,
            reward,
            right_hidden_state,
            return_prior=return_prior,
            sample=sample,
            detach_every=detach_every
        )
        # include prior gating values?
        gate_latent, gate_hidden_state = self.gating_network.encoder(
            action, state, value_errors[0], value_errors[1], gate_values[0], gate_values[1], gate_hidden_state
        )

        return (
            (gate_latent, gate_hidden_state),
            (left_latent_mean, left_latent_logvar, left_hidden_state),
            (right_latent_mean, right_latent_logvar, right_hidden_state)
        )

    def prior(self, num_processes):
        _, left_latent_mean, left_latent_logvar, left_hidden_state = self.left_actor_critic.encoder.prior(
            num_processes)
        _, right_latent_mean, right_latent_logvar, right_hidden_state = self.right_actor_critic.encoder.prior(
            num_processes)

        gate_latent, gate_hidden_state = self.gating_network.prior(
            num_processes)

        return (
            (gate_latent, gate_hidden_state),
            (left_latent_mean, left_latent_logvar, left_hidden_state),
            (right_latent_mean, right_latent_logvar, right_hidden_state)
        )

    def policy(self, state, latent, belief=None, task=None, deterministic=False):

        if isinstance(latent, tuple):
            gate_latent = latent[0]
            left_latent = latent[1]
            right_latent = latent[2]

        else:
            raise ValueError

        # get left hemisphere input to distribution
        left_value, left_actor_features = self.left_actor_critic.policy(
            state=state, latent=left_latent, belief=belief, task=task
        )

        # get right hemisphere input to distribution
        # NOTE: right hemisphere does not take state - will be converted to zeros
        right_value, right_actor_features = self.right_actor_critic.policy(
            state=state, latent=right_latent, belief=belief, task=task
        )

        # maybe gate network should take task? take combined latents and current state?
        left_gate_value, right_gate_value = self.gating_network.gating_function(
            gate_latent)

        combined_values, dist, actions = None, None, None
        chosen_hemisphere = None  # 0 = left, 1 = right, None = both

        left_action_mean = self.left_actor_critic.policy.dist.fc_mean(
            left_actor_features)

        right_action_mean = self.right_actor_critic.policy.dist.fc_mean(
            right_actor_features)

        if (self.gating_combination_method == GatingCombine.SUMMATION):
            # combine action and value estimate
            combined_action_means = left_gate_value * \
                left_action_mean + right_gate_value * right_action_mean
            combined_values = left_gate_value * left_value + right_gate_value * right_value

            # use 'self.std' for now
            std = torch.max(self.min_std, self.logstd.exp())
            dist = FixedNormal(combined_action_means, std)
            chosen_hemisphere = None  # both used
        elif (self.gating_combination_method == GatingCombine.SELECT_SAMPLE):
            # for preserving the computational graph and handling the two std devs
            std_devs = torch.stack((
                self.left_actor_critic.policy.dist.fc_logstd(
                    left_actor_features).exp(),
                self.right_actor_critic.policy.dist.fc_logstd(
                    right_actor_features).exp()
            ))

            softmax_std_devs = torch.softmax(std_devs, axis=0)
            # softmax represents the spread of distribution, therefore, lesser spread -> more confident
            hemisphere_confidence = 1 - softmax_std_devs

            gating_stack = torch.stack((
                left_gate_value,
                right_gate_value
            ))

            # finding confidence for the hemisphere per total action set
            confidence_values = gating_stack * hemisphere_confidence
            confidence_values = confidence_values.mean(dim=-1)
            combined_value_select_left_hemisphere = (
                confidence_values[0] > confidence_values[1]).unsqueeze(-1)
            select_left_hemisphere = combined_value_select_left_hemisphere.expand(
                    *combined_value_select_left_hemisphere.shape[:-1], 4)

            # finding distribution where the correct hemisphere is chosen
            combined_action_means = torch.where(
                select_left_hemisphere, left_action_mean, right_action_mean)
            combined_action_std_dev = torch.where(
                select_left_hemisphere, std_devs[0], std_devs[1])
            combined_action_std_dev = torch.clamp(
                combined_action_std_dev, min=self.min_std)
            dist = FixedNormal(combined_action_means, combined_action_std_dev)

            # track the number of times the hemisphere is chosen
            chosen_hemisphere = torch.zeros(2)
            chosen_hemisphere[0] = torch.count_nonzero(select_left_hemisphere)
            chosen_hemisphere[1] = torch.numel(
                select_left_hemisphere) - chosen_hemisphere[0]

            # combined value calculation
            left_combined = left_gate_value * left_value
            right_combined = right_gate_value * right_value
            combined_values = torch.where(
                combined_value_select_left_hemisphere, left_combined, right_combined)


        if deterministic:
            actions = dist.mean
        else:
            actions = dist.sample()  # assumes not deterministic

        assert (combined_values is not None) and (dist is not None) and (actions is not None), \
            'either combined_values, dist, or actions have not been set'

        return (combined_values, left_value, right_value), actions, dist, (left_gate_value, right_gate_value), chosen_hemisphere

    def act(self, state, latent, belief=None, task=None, deterministic=False):
        values, actions, _, gating_values, chosen_hemisphere = self.policy(
            state, latent, None, None, deterministic=deterministic)
        return values, actions, gating_values, chosen_hemisphere

    def get_value(self, state, latent, belief=None, task=None):
        value, _, _, _, _ = self.policy(state, latent, belief, task)
        return value

    def evaluate_actions(self, state, latent, belief, task, action):
        """
        Gets the distribution of the entire network
        """
        values, _, dist, gating_values, _ = self.policy(
            state, latent, None, None)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy, gating_values
