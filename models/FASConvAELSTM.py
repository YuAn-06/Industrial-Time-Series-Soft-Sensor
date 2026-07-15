"""FA-SConvAE-LSTM implementation for soft-sensor regression.

The model has two modes controlled by ``configs.model_stage``:

1. ``pretrain`` / ``pretrain_l*`` reconstructs ConvAE inputs.
2. ``finetune`` extracts spatial features, models time with an LSTM, and
   predicts the quality variable.
"""

import os

import torch
import torch.nn as nn


class FeatureAlignedConvBlock(nn.Module):
    """One feature-aligned convolutional encoder/decoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str,
    ):
        super().__init__()

        dilation = 4
        padding = dilation * (kernel_size - 1) // 2

        self.encoder = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            dilation=(1, dilation),
        )
        self.decoder = nn.ConvTranspose2d(
            out_channels,
            in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            dilation=(1, dilation),
        )

        # YAML activation names are lowercase, while PyTorch class names are not.
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        activation_name = activation.lower()
        if activation_name not in activations:
            raise NameError(
                f"Invalid activation name '{activation}'. Please check activation name"
            )
        self.activation = activations[activation_name]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one layer: [B, C_in, T, D] -> [B, C_out, T, D]."""
        return self.activation(self.encoder(x))

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """Decode one layer back to its input shape."""
        return self.activation(self.decoder(z))


class FeatureAlignedStackedConvAE(nn.Module):
    """Stack feature-aligned ConvAE blocks and flatten their output."""

    def __init__(
        self,
        input_dim: int,
        kernel_sizes: list[int],
        channels: list[int],
        activation: str,
    ):
        super().__init__()

        if len(kernel_sizes) != len(channels):
            raise ValueError("fa_kernel_sizes and fa_channels must have the same length.")
        if not kernel_sizes:
            raise ValueError("fa_kernel_sizes and fa_channels cannot be empty.")

        blocks = []
        in_channels = 1
        for kernel_size, out_channels in zip(kernel_sizes, channels):
            if kernel_size < 1:
                raise ValueError("All FA-SConvAE kernel sizes must be positive.")
            blocks.append(
                FeatureAlignedConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    activation,
                )
            )
            in_channels = out_channels

        self.blocks = nn.ModuleList(blocks)

        # Padding keeps feature width D unchanged. The last block therefore
        # produces channels[-1] * input_dim features at every time step.
        self.output_dim = channels[-1] * input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features: [B, T, D] -> [B, T, output_dim]."""
        z = x.unsqueeze(1)
        for block in self.blocks:
            z = block(z)
        z = z.permute(0, 2, 1, 3).contiguous()
        return z.flatten(start_dim=2)

    def pretrain(self, x: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return reconstruction targets for joint pretraining of all layers."""
        z = x.unsqueeze(1)
        outputs = []
        for block in self.blocks:
            target = z.detach()
            encoded = block(z)
            outputs.append({"rec": block.reconstruct(encoded), "target": target})
            z = encoded.detach()
        return outputs

    def pretrain_layer(
        self,
        x: torch.Tensor,
        layer_index: int,
    ) -> dict[str, torch.Tensor]:
        """Return the reconstruction pair for one layer-wise pretrain stage."""
        if not 0 <= layer_index < len(self.blocks):
            raise ValueError(f"Invalid pretrain layer index: {layer_index}.")

        z = x.unsqueeze(1)
        for index, block in enumerate(self.blocks):
            if index < layer_index:
                with torch.no_grad():
                    z = block(z)
                z = z.detach()
                continue

            target = z.detach()
            encoded = block(z)
            return {
                "rec": block.reconstruct(encoded),
                "target": target,
                "layer": layer_index,
            }

        raise RuntimeError("Unable to create the layer-wise pretraining output.")


class Model(nn.Module):
    """FA-SConvAE spatial encoder followed by an LSTM regressor."""

    def __init__(self, configs):
        super().__init__()

        if configs.task != "soft_sensor":
            raise ValueError("FASConvAELSTM supports only the soft_sensor task.")

        self.configs = configs
        self.stage = configs.model_stage

        self.spatial_encoder = FeatureAlignedStackedConvAE(
            input_dim=configs.enc_in,
            kernel_sizes=configs.fa_kernel_sizes,
            channels=configs.fa_channels,
            activation=configs.activation,
        )

        lstm_dropout = configs.dropout if configs.e_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.spatial_encoder.output_dim,
            hidden_size=configs.hidden_dim,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.projection = nn.Linear(configs.hidden_dim, configs.C_out)

        if configs.pretrained_ckpt:
            self.load_pretrained(configs.pretrained_ckpt)
            if configs.freeze_tae:
                self.freeze_spatial_encoder()

    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load only pretrained ConvAE weights, not LSTM/regression weights.

        Layer-wise pretraining checkpoints contain the complete model state
        because the experiment saves ``model.state_dict()``. The LSTM and
        projection are not trained during those stages, so loading them into
        fine-tuning is unnecessary and would prevent changing ``hidden_dim``.
        """
        state = torch.load(os.path.normpath(checkpoint_path), map_location="cpu")
        state = state.get("model", state)
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}

        prefix = "spatial_encoder."
        spatial_state = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not spatial_state:
            raise KeyError(
                f"No '{prefix}*' weights found in checkpoint: {checkpoint_path}"
            )
        return self.spatial_encoder.load_state_dict(spatial_state, strict=strict)

    def freeze_spatial_encoder(self):
        """Prevent the pretrained ConvAE weights from changing in fine-tuning."""
        for parameter in self.spatial_encoder.parameters():
            parameter.requires_grad = False

    def _pretrain_layer_index(self):
        """Translate a layer-wise stage name to a zero-based block index."""
        layer_map = {
            "pretrain_l1": 0,
            "pretrain_l2": 1,
            "pretrain_l3": 2,
        }
        return layer_map.get(self.stage)

    def optimizer_param_groups(self, base_lr: float, weight_decay: float):
        """Give each ConvAE layer its configured pretraining learning rate."""
        if not self.stage.startswith("pretrain"):
            return None

        learning_rates = self.configs.fa_pretrain_learning_rates
        if len(learning_rates) != len(self.spatial_encoder.blocks):
            raise ValueError(
                "fa_pretrain_learning_rates must match the number of ConvAE layers."
            )

        layer_index = self._pretrain_layer_index()
        if layer_index is not None:
            block = self.spatial_encoder.blocks[layer_index]
            learning_rate = learning_rates[layer_index]
            return [{
                "params": block.parameters(),
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "weight_decay": weight_decay,
            }]

        return [
            {
                "params": block.parameters(),
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "weight_decay": weight_decay,
            }
            for block, learning_rate in zip(
                self.spatial_encoder.blocks,
                learning_rates,
            )
        ]

    def pretrain_forward(self, x_enc: torch.Tensor):
        """Build the reconstruction output consumed by FASConvAELSTM_Loss."""
        layer_index = self._pretrain_layer_index()
        if layer_index is None:
            output = self.spatial_encoder.pretrain(x_enc)
        else:
            output = self.spatial_encoder.pretrain_layer(x_enc, layer_index)
        return {"fa_sconvae": output}

    def finetune_forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Run ConvAE -> LSTM -> regression projection."""
        spatial_features = self.spatial_encoder(x_enc)
        temporal_features, _ = self.lstm(spatial_features)
        return self.projection(temporal_features[:, -1, :])

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        batch_y,
        flag="train",
    ):
        """Dispatch to pretraining or fine-tuning according to model_stage."""
        if self.stage.startswith("pretrain"):
            return self.pretrain_forward(x_enc)
        return self.finetune_forward(x_enc)
