"""
temporal_a2c_network.py -- GRU / Transformer PPO temporal baselines.

This builder mirrors the GLA policy interface but replaces the temporal
encoder over fixed observation-history tokens with either a GRU or a standard
TransformerEncoder. It is intended for fair main-table baselines: same task,
same history_observation layout, same PPO pipeline, different sequence model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class TemporalA2CBuilder(NetworkBuilder):
    """rl_games NetworkBuilder for GRU/Transformer temporal PPO baselines."""

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return TemporalA2CBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self._load_params(params)

            assert len(input_shape) == 1, (
                f"TemporalA2C expects flat observations, got input_shape={input_shape}"
            )
            obs_dim = int(input_shape[0])
            history_size = self.history_length * self.token_dim
            aux_size = self.aux_target_dim if self.aux_enabled else 0
            assert obs_dim > history_size + aux_size, (
                f"obs_dim={obs_dim} must be larger than history block "
                f"history_length*token_dim={history_size} + aux_target_dim={aux_size}"
            )
            self.obs_dim = obs_dim
            self.aux_size = aux_size
            self.base_dim = obs_dim - history_size - aux_size

            mlp_args = {
                "input_size": self.base_dim,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            base_out = self.units[-1] if len(self.units) > 0 else self.base_dim

            self.token_proj = nn.Linear(self.token_dim, self.temporal_hidden)
            if self.encoder == "gru":
                self.temporal_encoder = nn.GRU(
                    input_size=self.temporal_hidden,
                    hidden_size=self.temporal_hidden,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=self.dropout if self.num_layers > 1 else 0.0,
                )
                self.pos_embed = None
            elif self.encoder == "transformer":
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.history_length, self.temporal_hidden)
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.temporal_hidden,
                    nhead=self.num_heads,
                    dim_feedforward=self.ffn_hidden,
                    dropout=self.dropout,
                    activation=self.transformer_activation,
                    batch_first=True,
                    norm_first=True,
                )
                self.temporal_encoder = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=self.num_layers,
                )
            else:
                raise ValueError(f"Unsupported temporal encoder: {self.encoder!r}")
            self.temporal_norm = nn.LayerNorm(self.temporal_hidden)

            fused_dim = base_out + self.temporal_hidden
            out_size = fused_dim

            self.value = nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            assert self.is_continuous, "TemporalA2C only supports continuous actions."
            self.mu = nn.Linear(out_size, actions_num)
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
                self.sigma = nn.Linear(out_size, actions_num)

            if self.aux_enabled and self.aux_pred_dim > 0:
                self.aux_head = nn.Sequential(
                    nn.Linear(self.temporal_hidden, self.aux_hidden),
                    nn.ELU(),
                    nn.Linear(self.aux_hidden, self.aux_pred_dim),
                )
            else:
                self.aux_head = None
            self.last_aux_loss = None
            self.last_aux_loss_components = {}
            if self.aux_enabled and self.aux_pred_dim > 0:
                weights = torch.tensor(self.aux_target_weights, dtype=torch.float32)
                self.register_buffer("_aux_weights_buf", weights, persistent=False)
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
            if self.pos_embed is not None:
                nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
            for module in self.modules():
                if isinstance(module, nn.GRU):
                    for name, param in module.named_parameters():
                        if "weight" in name:
                            nn.init.xavier_uniform_(param)
                        elif "bias" in name:
                            nn.init.zeros_(param)
            mu_init(self.mu.weight)
            if self.space_config["fixed_sigma"]:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

            print(
                f"  [temporal] encoder={self.encoder} obs_dim={obs_dim} "
                f"base_dim={self.base_dim} history={self.history_length}x{self.token_dim} "
                f"hidden={self.temporal_hidden} layers={self.num_layers} "
                f"heads={self.num_heads if self.encoder == 'transformer' else '-'} "
                f"pool={self.pool} aux_enabled={self.aux_enabled} "
                f"aux_pred_dim={self.aux_pred_dim} aux_target_dim={self.aux_target_dim} "
                f"aux_target_keys={self.aux_target_keys}"
            )

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            is_train = obs_dict.get("is_train", True)
            batch = obs.shape[0]

            if self.aux_enabled and self.aux_size > 0:
                aux_targets = obs[:, -self.aux_size:]
                obs_main = obs[:, :-self.aux_size]
            else:
                aux_targets = None
                obs_main = obs

            base_obs = obs_main[:, : self.base_dim]
            history_flat = obs_main[:, self.base_dim:]
            history_tokens = history_flat.view(
                batch,
                self.history_length,
                self.token_dim,
            )

            base_feat = self.actor_mlp(base_obs)
            tokens = self.token_proj(history_tokens)
            if self.encoder == "gru":
                temporal_out, _ = self.temporal_encoder(tokens)
            else:
                scale = math.sqrt(float(self.temporal_hidden))
                temporal_out = self.temporal_encoder(tokens * scale + self.pos_embed)

            if self.pool == "mean":
                temporal_feat = temporal_out.mean(dim=1)
            else:
                temporal_feat = temporal_out[:, -1, :]
            temporal_feat = self.temporal_norm(temporal_feat)

            fused = torch.cat([base_feat, temporal_feat], dim=-1)
            value = self.value_act(self.value(fused))

            mu = self.mu_act(self.mu(fused))
            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(fused))

            if (
                self.aux_enabled
                and self.aux_head is not None
                and is_train
                and aux_targets is not None
            ):
                aux_pred = self.aux_head(temporal_feat)
                target_used = aux_targets[:, : self.aux_pred_dim]
                per_ch_mse = (aux_pred - target_used).pow(2).mean(dim=0)
                self.last_aux_loss = (per_ch_mse * self._aux_weights_buf).sum()
                self.last_aux_loss_components = {
                    key: per_ch_mse[i].detach()
                    for i, key in enumerate(self.aux_target_keys)
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
                "TemporalA2C expects a continuous action space."
            )
            self.is_continuous = True
            self.space_config = params["space"]["continuous"]

            temporal_cfg = params.get("temporal", {})
            self.encoder = str(temporal_cfg.get("encoder", "gru")).lower()
            self.history_length = int(temporal_cfg.get("history_length", 16))
            self.token_dim = int(temporal_cfg.get("token_dim", 102))
            self.temporal_hidden = int(temporal_cfg.get("hidden_size", 128))
            self.num_layers = int(temporal_cfg.get("num_layers", 1))
            self.num_heads = int(temporal_cfg.get("num_heads", 4))
            self.ffn_hidden = int(
                temporal_cfg.get("ffn_hidden", self.temporal_hidden * 4)
            )
            self.dropout = float(temporal_cfg.get("dropout", 0.0))
            self.transformer_activation = str(
                temporal_cfg.get("activation", "gelu")
            )
            self.pool = str(temporal_cfg.get("pool", "last")).lower()
            assert self.encoder in ("gru", "transformer"), (
                f"temporal.encoder must be 'gru' or 'transformer', got {self.encoder!r}"
            )
            assert self.pool in ("last", "mean"), (
                f"temporal.pool must be 'last' or 'mean', got {self.pool!r}"
            )
            if self.encoder == "transformer":
                assert self.temporal_hidden % self.num_heads == 0, (
                    f"hidden_size={self.temporal_hidden} must be divisible by "
                    f"num_heads={self.num_heads}"
                )

            aux = params.get("phys_aux", {}) or {}
            self.aux_enabled = bool(aux.get("enabled", False))
            self.aux_pred_dim = int(aux.get("pred_dim", 0))
            self.aux_target_dim = int(aux.get("target_dim", self.aux_pred_dim))
            self.aux_hidden = int(aux.get("hidden_size", 64))

            cfg_keys = list(aux.get("target_keys") or [])
            if not cfg_keys and self.aux_enabled:
                cfg_keys = ["dq_obj", "slip_proxy", "tracking_stress"][: self.aux_pred_dim]
            self.aux_target_keys = cfg_keys

            cfg_weights = list(aux.get("target_weights") or [])
            if len(cfg_weights) != self.aux_pred_dim:
                cfg_weights = [1.0] * self.aux_pred_dim
            self.aux_target_weights = [float(weight) for weight in cfg_weights]


def register_temporal_network():
    """Register the temporal baseline builder before Runner.load()."""
    from rl_games.algos_torch import model_builder

    model_builder.register_network("temporal_actor_critic", TemporalA2CBuilder)
