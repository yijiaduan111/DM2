"""
pica_mlp_a2c_network.py -- no-GLA PICA auxiliary ablation network.

This builder keeps the PICA physical auxiliary head/loss interface used by
`pica_a2c_agent.py`, but removes the GLA temporal encoder, temporal latent,
sequence aggregation, and long-history policy encoder.

The policy/value path is a plain MLP over the non-aux observation channels.
The auxiliary head also reads this MLP feature and predicts the physical
auxiliary targets appended at the observation tail by HandDragTask.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class PICAMLPA2CBuilder(NetworkBuilder):
    """rl_games NetworkBuilder for PICA no-GLA MLP policy + aux head."""

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return PICAMLPA2CBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self._load_params(params)

            assert len(input_shape) == 1, (
                f"PICAMLPA2C expects flat observations, got input_shape={input_shape}"
            )
            obs_dim = int(input_shape[0])
            self.obs_dim = obs_dim
            self.aux_size = self.aux_target_dim if self.aux_enabled else 0
            assert obs_dim > self.aux_size, (
                f"obs_dim={obs_dim} must be larger than aux_target_dim={self.aux_size}"
            )
            self.policy_obs_dim = obs_dim - self.aux_size

            mlp_args = {
                "input_size": self.policy_obs_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            feat_dim = self.units[-1] if len(self.units) > 0 else self.policy_obs_dim

            self.value = nn.Linear(feat_dim, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            assert self.is_continuous, (
                "PICAMLPA2C only supports continuous action space currently."
            )
            self.mu = nn.Linear(feat_dim, actions_num)
            self.mu_act = self.activations_factory.create(
                self.space_config["mu_activation"]
            )
            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            self.sigma_act = self.activations_factory.create(
                self.space_config["sigma_activation"]
            )
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

            if self.space_config["fixed_sigma"]:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = nn.Linear(feat_dim, actions_num)

            if self.aux_enabled and self.aux_pred_dim > 0:
                self.aux_head = nn.Sequential(
                    nn.Linear(feat_dim, self.aux_hidden),
                    nn.ELU(),
                    nn.Linear(self.aux_hidden, self.aux_pred_dim),
                )
            else:
                self.aux_head = None
            self.last_aux_loss = None
            self.last_aux_loss_components = {}

            if self.aux_enabled and self.aux_pred_dim > 0:
                self.register_buffer(
                    "_aux_weights_buf",
                    torch.tensor(self.aux_target_weights, dtype=torch.float32),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "_aux_weights_buf",
                    torch.zeros(1, dtype=torch.float32),
                    persistent=False,
                )

            mlp_init = self.init_factory.create(**self.initializer)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    mlp_init(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
            mu_init(self.mu.weight)
            if self.space_config["fixed_sigma"]:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

            print(
                f"  [pica_mlp] obs_dim={obs_dim} policy_obs_dim={self.policy_obs_dim} "
                f"aux_enabled={self.aux_enabled} aux_pred_dim={self.aux_pred_dim} "
                f"aux_target_dim={self.aux_target_dim} aux_mode={self.aux_mode} "
                f"aux_target_keys={self.aux_target_keys}"
            )

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            is_train = obs_dict.get("is_train", True)

            if self.aux_enabled and self.aux_size > 0:
                aux_targets = obs[:, -self.aux_size:]
                policy_obs = obs[:, :-self.aux_size]
            else:
                aux_targets = None
                policy_obs = obs

            feat = self.actor_mlp(policy_obs)
            value = self.value_act(self.value(feat))
            mu = self.mu_act(self.mu(feat))
            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(feat))

            if (
                self.aux_enabled
                and self.aux_head is not None
                and is_train
                and aux_targets is not None
            ):
                aux_pred = self.aux_head(feat)
                target_used = aux_targets[:, : self.aux_pred_dim]
                per_ch_mse = (aux_pred - target_used).pow(2).mean(dim=0)
                self.last_aux_loss = (per_ch_mse * self._aux_weights_buf).sum()
                self.last_aux_loss_components = {
                    key: per_ch_mse[idx].detach()
                    for idx, key in enumerate(self.aux_target_keys)
                }
            else:
                self.last_aux_loss = torch.zeros((), device=obs.device)
                self.last_aux_loss_components = {}

            return mu, sigma, value, states

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def _load_params(self, params):
            mlp = params.get("mlp", {})
            self.units = mlp.get("units", [256, 256])
            self.activation = mlp.get("activation", "elu")
            self.initializer = mlp.get("initializer", {"name": "default"})
            self.is_d2rl = mlp.get("d2rl", False)
            self.norm_only_first_layer = mlp.get("norm_only_first_layer", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)

            assert "space" in params and "continuous" in params["space"], (
                "PICAMLPA2C expects a continuous action space."
            )
            self.is_continuous = True
            self.space_config = params["space"]["continuous"]

            aux = params.get("phys_aux", {}) or {}
            self.aux_enabled = bool(aux.get("enabled", False))
            self.aux_pred_dim = int(aux.get("pred_dim", 0))
            self.aux_target_dim = int(aux.get("target_dim", self.aux_pred_dim))
            self.aux_hidden = int(aux.get("hidden_size", 64))
            self.aux_mode = str(aux.get("mode", "current"))

            cfg_keys = list(aux.get("target_keys") or [])
            if not cfg_keys and self.aux_enabled:
                cfg_keys = ["dq_obj", "slip_proxy", "tracking_stress"][: self.aux_pred_dim]
            self.aux_target_keys = cfg_keys

            cfg_weights = list(aux.get("target_weights") or [])
            if len(cfg_weights) != self.aux_pred_dim:
                cfg_weights = [1.0] * self.aux_pred_dim
            self.aux_target_weights = [float(weight) for weight in cfg_weights]


def register_pica_mlp_network():
    """Register the no-GLA PICA MLP network builder."""
    from rl_games.algos_torch import model_builder

    model_builder.register_network("pica_mlp_actor_critic", PICAMLPA2CBuilder)
