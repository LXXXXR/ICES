# From https://github.com/oxwhirl/wqmix/
# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer


import pyro
from pyro.infer import SVI, Trace_ELBO
import torch.nn.functional as F
import torch as th
import torch.nn as nn
from torch.optim import RMSprop, Adam
import numpy as np

from modules.ices.state_pred_cvae import StatePredCVAE, StatePredBL
from modules.agents.ices_agent import ICESCritic
from utils.helper_func import KL_div, get_gard_norm


class ICES_DMAQ_qattenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.int_params = list(mac.int_parameters())

        self.last_target_update_episode = 0
        self.pred_s_len = args.pred_s_len

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
            elif args.mixer == "dmaq_qatten":
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.int_critic = ICESCritic(
            mac._get_input_shape(scheme) + int(np.prod(self.args.state_shape)), args
        )

        self.world_bl_global = StatePredBL(args)
        self.world_bl_local = StatePredBL(args)
        self.world_model = StatePredCVAE(
            args,
            baseline_net_global=self.world_bl_global,
            baseline_net_local=self.world_bl_local,
        )

        if self.args.use_cuda:
            self.world_bl_global.to(th.device(self.args.GPU))
            self.world_bl_local.to(th.device(self.args.GPU))
            self.world_model.to(th.device(self.args.GPU))
            self.int_critic.to(th.device(self.args.GPU))

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        self.int_optimiser = RMSprop(
            params=self.int_params,
            lr=args.lr,
            alpha=args.optim_alpha,
            eps=args.optim_eps,
        )
        self.int_c_optimiser = RMSprop(
            params=self.int_critic.parameters(),
            lr=args.lr,
            alpha=args.optim_alpha,
            eps=args.optim_eps,
        )
        self.world_bl_optimizer_global = th.optim.Adam(
            self.world_model.baseline_net_global.parameters(),
            lr=args.world_bl_lr,
            eps=args.optim_eps,
            weight_decay=args.weight_decay,
        )
        self.world_bl_optimizer_local = th.optim.Adam(
            self.world_model.baseline_net_local.parameters(),
            lr=args.world_bl_lr,
            eps=args.optim_eps,
            weight_decay=args.weight_decay,
        )
        num_steps = args.t_max // args.env_args["time_limit"] * self.pred_s_len
        lrd = args.world_gamma ** (1 / num_steps)

        self.world_optimizer_global = pyro.optim.ClippedAdam(
            {
                "lr": args.world_lr,
                "betas": (0.95, 0.999),
                "lrd": lrd,
                "clip_norm": args.world_clip_param,
            }
        )
        self.world_optimizer_local = pyro.optim.ClippedAdam(
            {
                "lr": args.world_lr,
                "betas": (0.95, 0.999),
                "lrd": lrd,
                "clip_norm": args.world_clip_param,
            }
        )

        self.world_svi_global = SVI(
            self.world_model.model_global,
            self.world_model.guide_global,
            self.world_optimizer_global,
            loss=Trace_ELBO(),
        )
        self.world_svi_local = SVI(
            self.world_model.model_local,
            self.world_model.guide_local,
            self.world_optimizer_local,
            loss=Trace_ELBO(),
        )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.world_log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

        self.world_flag = True

    def train_world(self, batch: EpisodeBatch, t_env: int):
        # update two model iteratively
        n_mini = self.pred_s_len
        idxes = np.arange(1, n_mini + 1)
        # ranom order of [1,2,...,n_mini]
        np.random.shuffle(idxes)

        # (bs, seq_len, s_dim)
        s = batch["state"][:, :-1, :]
        # (bs, seq_len, n_agents)
        a = batch["actions"][:, :-1, :, 0].long()
        n_agents = a.shape[-1]

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.world_flag:
            (
                pred_bl_loss_global_log,
                world_bl_grad_norm_global_log,
                pred_loss_global_log,
                world_grad_norm_log,
            ) = (0, 0, 0, 0)

            for idx in idxes:
                s_prime = batch["state"][:, idx:, :]
                bs, seq_len_cut, _ = s_prime.shape
                s_ori = s[:, :seq_len_cut, :]
                s_diff = (s_prime - s_ori).reshape(bs * seq_len_cut, -1)
                s_ori = s_ori.reshape(bs * seq_len_cut, -1)
                a_all = a[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)
                mask_sample = mask[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)

                s_diff_pred = self.world_model.baseline_net_global(s_ori, a_all)
                pred_bl_loss_global = F.mse_loss(
                    s_diff.detach() * mask_sample, s_diff_pred * mask_sample
                )
                self.world_bl_optimizer_global.zero_grad()
                pred_bl_loss_global.backward()
                world_bl_grad_norm_global = nn.utils.clip_grad_norm_(
                    self.world_model.baseline_net_global.parameters(),
                    self.args.grad_norm_clip,
                )
                self.world_bl_optimizer_global.step()

                # learn prediction network
                world_grad_norm = get_gard_norm(self.world_model.parameters())
                pred_loss_global = self.world_svi_global.step(
                    s_ori * mask_sample,
                    (a_all * mask_sample).long(),
                    s_prime=s_diff.detach() * mask_sample,
                )

                pred_bl_loss_global_log += pred_bl_loss_global.item()
                world_bl_grad_norm_global_log += world_bl_grad_norm_global.item()
                pred_loss_global_log += pred_loss_global
                world_grad_norm_log += world_grad_norm

            if t_env - self.world_log_stats_t >= self.args.learner_log_interval:
                pred_bl_loss_global_log /= n_mini
                world_bl_grad_norm_global_log /= n_mini
                pred_loss_global_log /= n_mini
                world_grad_norm_log /= n_mini
                self.logger.log_stat(
                    "pred_bl_loss_global", pred_bl_loss_global_log, t_env
                )
                self.logger.log_stat(
                    "world_bl_grad_norm_global", world_bl_grad_norm_global_log, t_env
                )
                self.logger.log_stat("pred_loss_global", pred_loss_global_log, t_env)
                self.logger.log_stat("world_grad_norm", world_grad_norm_log, t_env)
                self.world_log_stats_t = t_env

        else:
            (
                pred_bl_loss_local_log,
                world_bl_grad_norm_local_log,
                pred_loss_local_log,
                world_grad_norm_log,
            ) = (0, 0, 0, 0)

            for idx in idxes:
                s_prime = batch["state"][:, idx:, :]
                bs, seq_len_cut, _ = s_prime.shape
                s_ori = s[:, :seq_len_cut, :]
                s_diff = (s_prime - s_ori).reshape(bs * seq_len_cut, -1)
                s_ori = s_ori.reshape(bs * seq_len_cut, -1)
                a_all = a[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)
                mask_sample = mask[:, :seq_len_cut, :].reshape(bs * seq_len_cut, -1)
                a_masked = a_all.clone()
                a_mask = th.ones_like(a_masked)
                one_pos = th.randint(n_agents, (bs * seq_len_cut, 1)).to(
                    self.args.device
                )
                a_mask.scatter_(1, index=one_pos, src=th.zeros_like(a_mask))
                a_masked = ((a_masked + 1) * a_mask - 1).long()

                s_diff_pred = self.world_model.baseline_net_local(s_ori, a_masked)
                pred_bl_loss_local = F.mse_loss(
                    s_diff.detach() * mask_sample, s_diff_pred * mask_sample
                )
                self.world_bl_optimizer_local.zero_grad()
                pred_bl_loss_local.backward()
                world_bl_grad_norm_local = nn.utils.clip_grad_norm_(
                    self.world_model.baseline_net_local.parameters(),
                    self.args.grad_norm_clip,
                )
                self.world_bl_optimizer_local.step()

                # learn prediction network
                world_grad_norm = get_gard_norm(self.world_model.parameters())
                pred_loss_local = self.world_svi_local.step(
                    s_ori * mask_sample,
                    (a_masked * mask_sample).long(),
                    s_prime=s_diff.detach() * mask_sample,
                )

                pred_bl_loss_local_log += pred_bl_loss_local.item()
                world_bl_grad_norm_local_log += world_bl_grad_norm_local.item()
                pred_loss_local_log += pred_loss_local
                world_grad_norm_log += world_grad_norm

            if t_env - self.world_log_stats_t >= self.args.learner_log_interval:
                pred_bl_loss_local_log /= n_mini
                world_bl_grad_norm_local_log /= n_mini
                pred_loss_local_log /= n_mini
                world_grad_norm_log /= n_mini
                self.logger.log_stat(
                    "pred_bl_loss_local", pred_bl_loss_local_log, t_env
                )
                self.logger.log_stat(
                    "world_bl_grad_norm_local", world_bl_grad_norm_local_log, t_env
                )
                self.logger.log_stat("pred_loss_local", pred_loss_local_log, t_env)
                self.logger.log_stat("world_grad_norm", world_grad_norm_log, t_env)
                self.world_log_stats_t = t_env

        self.world_flag = not self.world_flag

    def sub_train(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        mac,
        mixer,
        optimiser,
        params,
        show_demo=False,
        save_data=None,
    ):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat(
            [th.zeros_like(actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1
        )  # last_actions

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(-1, initial_hidden.shape[-1]).to(
            self.args.device
        )
        input_here = (
            th.cat((batch["obs"], last_actions_onehot), dim=-1)
            .permute(0, 2, 1, 3)
            .to(self.args.device)
        )

        mac_out, hidden_store, _ = self.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach()
        )
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3
        )  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]
        ).to(self.args.device)
        target_mac_out, _, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach()
        )
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(
                3
            )
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(
                cur_max_actions.squeeze(3).shape + (self.n_actions,)
            ).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(
                3, cur_max_actions, 1
            )
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if mixer is not None:
            if self.args.mixer == "dmaq_qatten":
                ans_chosen, q_attend_regs, head_entropies = mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    is_v=True,
                    obs=batch["obs"][:, :-1],
                )
                ans_adv, _, _ = mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    actions=actions_onehot,
                    max_q_i=max_action_qvals,
                    is_v=False,
                    obs=batch["obs"][:, :-1],
                )
                chosen_action_qvals = ans_chosen + ans_adv
            else:
                ans_chosen = mixer(
                    chosen_action_qvals, batch["state"][:, :-1], is_v=True
                )
                ans_adv = mixer(
                    chosen_action_qvals,
                    batch["state"][:, :-1],
                    actions=actions_onehot,
                    max_q_i=max_action_qvals,
                    is_v=False,
                )
                chosen_action_qvals = ans_chosen + ans_adv

            if self.args.double_q:
                if self.args.mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mixer(
                        target_chosen_qvals,
                        batch["state"][:, 1:],
                        is_v=True,
                        obs=batch["obs"][:, :-1],
                    )
                    target_adv, _, _ = self.target_mixer(
                        target_chosen_qvals,
                        batch["state"][:, 1:],
                        actions=cur_max_actions_onehot,
                        max_q_i=target_max_qvals,
                        is_v=False,
                        obs=batch["obs"][:, :-1],
                    )
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mixer(
                        target_chosen_qvals, batch["state"][:, 1:], is_v=True
                    )
                    target_adv = self.target_mixer(
                        target_chosen_qvals,
                        batch["state"][:, 1:],
                        actions=cur_max_actions_onehot,
                        max_q_i=target_max_qvals,
                        is_v=False,
                    )
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(
                    target_max_qvals, batch["state"][:, 1:], is_v=True
                )

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print(
                "action_pair_%d_%d" % (save_data[0], save_data[1]),
                np.squeeze(q_data[:, 0]),
                np.squeeze(q_i_data[:, 0]),
                np.squeeze(tot_q_data[:, 0]),
                np.squeeze(tot_target[:, 0]),
            )
            self.logger.log_stat(
                "action_pair_%d_%d" % (save_data[0], save_data[1]),
                np.squeeze(tot_q_data[:, 0]),
                t_env,
            )
            return

        # Td-error
        td_error = chosen_action_qvals - targets.detach()

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.args.mixer == "dmaq_qatten":
            loss = (masked_td_error**2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error**2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        ########## intrinsic
        # Calculate estimated Q-Values
        int_c_initial_hidden = (
            self.int_critic.init_hidden()
            .unsqueeze(0)
            .expand(batch.batch_size, self.args.n_agents, -1)
        )
        int_c_initial_hidden = int_c_initial_hidden.clone().detach()
        int_c_initial_hidden = int_c_initial_hidden.reshape(
            -1, int_c_initial_hidden.shape[-1]
        ).to(self.args.device)
        int_initial_hidden = self.mac.int_hidden_states.clone().detach()
        int_initial_hidden = int_initial_hidden.reshape(
            -1, int_initial_hidden.shape[-1]
        ).to(self.args.device)
        input_here = (
            th.cat(
                (
                    batch["obs"],
                    batch["state"][:, :, None, :].expand(
                        -1, -1, self.args.n_agents, -1
                    ),
                    last_actions_onehot,
                ),
                dim=-1,
            )
            .permute(0, 2, 1, 3)
            .to(self.args.device)
        )
        int_mac_out, _, _ = self.mac.int_agent.forward(
            input_here.clone().detach(), int_initial_hidden.clone().detach()
        )
        int_v, _, _ = self.int_critic.forward(
            input_here.clone().detach(), int_c_initial_hidden.clone().detach()
        )
        int_v = int_v[:, :-1, :, 0]

        int_mac_out_pi = F.log_softmax(int_mac_out, dim=-1)
        int_entropy = -th.sum(int_mac_out_pi * F.softmax(int_mac_out, dim=-1), dim=-1)[
            :, :-1
        ]
        int_mac_out_pi = th.gather(
            int_mac_out_pi[:, :-1], dim=3, index=actions
        ).squeeze(3)

        bs, seq_len, n_agents, _ = actions.shape
        # (bs, seq_len, s_dim) -> (bs * seq_len, s_dim)
        s = batch["state"][:, :-1, :].reshape(bs * seq_len, -1)
        # (bs, seq_len, n_agents, 1) -> (bs * seq_len, n_agents)
        a_all = actions.reshape(bs * seq_len, n_agents).long()
        s_expand = (
            batch["state"][:, :-1, None, :]
            .expand(-1, -1, n_agents, -1)
            .reshape(bs * seq_len * n_agents, -1)
        )
        # (bs, seq_len, n_agents, n_agents)
        a_masked = actions[:, :, None, :, 0].expand(-1, -1, n_agents, -1)
        a_masked = (a_masked + 1) * (1.0 - th.eye(n_agents).to(self.args.device))[
            None, None, :, :
        ]
        a_masked = (a_masked - 1).long().reshape(bs * seq_len * n_agents, -1)

        with th.no_grad():
            global_mu, global_sigma, _, _ = self.world_model.model_global(s, a_all)
            local_mu, local_sigma, _, _ = self.world_model.model_local(
                s_expand, a_masked
            )
            global_mu = global_mu.reshape(bs * seq_len, 1, -1)
            global_sigma = global_sigma.reshape(bs * seq_len, 1, -1)
            local_mu = local_mu.reshape(bs * seq_len, n_agents, -1)
            local_sigma = local_sigma.reshape(bs * seq_len, n_agents, -1)
            div = KL_div(global_mu, global_sigma, local_mu, local_sigma)
            # normalize in an ugly way
            div = 1.0 - th.exp(-div)

            int_rewards = div.reshape(
                bs,
                seq_len,
                n_agents,
            )

            adv = int_rewards - int_v

        # Td-error
        mask = mask.expand_as(adv)
        # 0-out the targets that came from padded data
        masked_adv = adv * mask
        mask_entropy = int_entropy * mask
        # Normal L2 loss, take mean over actual data
        # int_loss value too small
        int_loss = (
            -10
            * (
                (masked_adv * int_mac_out_pi).sum()
                + self.args.int_ent_coef * mask_entropy.sum()
            )
            / mask.sum()
        )

        # Optimise
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        self.int_optimiser.zero_grad()
        int_loss.backward()
        int_grad_norm = th.nn.utils.clip_grad_norm_(
            self.int_params, self.args.grad_norm_clip
        )
        self.int_optimiser.step()

        int_error = (int_v - int_rewards) * mask
        loss_int_c = 10 * (int_error**2).sum() / mask.sum()
        self.int_c_optimiser.zero_grad()
        loss_int_c.backward()
        int_c_grad_norm = th.nn.utils.clip_grad_norm_(
            self.int_critic.parameters(), self.args.grad_norm_clip
        )
        self.int_c_optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("int_loss", int_loss.item(), t_env)
            self.logger.log_stat("loss_int_c", loss_int_c.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("int_grad_norm", int_grad_norm, t_env)
            self.logger.log_stat("int_c_grad_norm", int_c_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "intrinsic_reward_mean", int_rewards.mean().item(), t_env
            )
            self.logger.log_stat(
                "intrinsic_reward_std", int_rewards.std().item(), t_env
            )
            self.log_stats_t = t_env

    def train(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        show_demo=False,
        save_data=None,
    ):
        self.sub_train(
            batch,
            t_env,
            episode_num,
            self.mac,
            self.mixer,
            self.optimiser,
            self.params,
            show_demo=show_demo,
            save_data=save_data,
        )
        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        self.int_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.int_critic.state_dict(), "{}/int_c.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        th.save(self.int_optimiser.state_dict(), "{}/int_opt.th".format(path))
        th.save(self.int_c_optimiser.state_dict(), "{}/c_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
            self.target_mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.int_critic.load_state_dict(
            th.load(
                "{}/int_c.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.int_optimiser.load_state_dict(
            th.load(
                "{}/int_opt.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        self.int_c_optimiser.load_state_dict(
            th.load(
                "{}/c_opt.th".format(path), map_location=lambda storage, loc: storage
            )
        )
