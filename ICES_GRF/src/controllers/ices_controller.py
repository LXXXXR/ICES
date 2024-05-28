import torch as th
import numpy as np

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class ICESMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(ICESMAC, self).__init__(scheme, groups, args)

        self.int_hidden_states = None

        self.t_max = args.t_max
        self.int_ratio = args.int_ratio
        self.int_finish = args.int_finish

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        int_agent_outputs = self.int_forward(ep_batch, t_ep, test_mode=test_mode)
        if test_mode:
            int_ratio = 0.0
            log_info = {}
        else:
            int_ratio = self.int_ratio
        chosen_actions, int_entropy = self.action_selector.select_action(
            agent_outputs[bs],
            int_agent_outputs[bs],
            avail_actions[bs],
            t_env,
            int_ratio,
            test_mode=test_mode,
        )
        if not test_mode:
            log_info = {
                "ext_value_estimation_m": agent_outputs.detach().mean().item(),
                "ext_value_estimation_v": agent_outputs.detach().var().item(),
                "int_entropy": int_entropy.detach().mean().item(),
                "int_ratio": int_ratio,
            }
        return chosen_actions, log_info

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states, _ = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def int_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_int_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        reshaped_avail_actions = avail_actions.reshape(
            ep_batch.batch_size * self.n_agents, -1
        )
        int_agent_outs, self.int_hidden_states, _ = self.int_agent(
            agent_inputs, self.int_hidden_states
        )
        int_agent_outs[reshaped_avail_actions == 0] = -float("inf")
        return int_agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )  # bav
        self.int_hidden_states = (
            self.int_agent.init_hidden()
            .unsqueeze(0)
            .expand(batch_size, self.n_agents, -1)
        )  # bav

    def parameters(self):
        return self.agent.parameters()

    def int_parameters(self):
        return self.int_agent.parameters()

    def load_state(self, other_mac, agent_flag=True, int_agent_flag=True):
        if agent_flag:
            self.agent.load_state_dict(other_mac.agent.state_dict())
        if int_agent_flag:
            self.int_agent.load_state_dict(other_mac.int_agent.state_dict())

    def cuda(self):
        self.agent.to(self.args.GPU)
        self.int_agent.to(self.args.GPU)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.int_agent.state_dict(), "{}/int_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.int_agent.load_state_dict(
            th.load(
                "{}/int_agent.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](
            input_shape, self.args, nonlinear=False
        )
        self.int_agent = agent_REGISTRY[self.args.agent](
            input_shape + int(np.prod(self.args.state_shape)),
            self.args,
            nonlinear=False,
        )

    def _build_int_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        inputs.append(batch["state"][:, t][:, None, :].expand(-1, self.n_agents, -1))
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs
