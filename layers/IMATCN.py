import torch
import torch.nn as nn


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-2):
        super(FRN, self).__init__()
        if isinstance(num_features, (list, tuple)):
            num_features = num_features[-1]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((1, num_features, 1)))
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1)))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.4):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.prelu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.prelu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.prelu = nn.PReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.prelu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.4):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )
            layers.append(nn.LeakyReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNs(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNs, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.frn = FRN(num_channels)

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        y = self.tcn(inputs)
        return self.frn(y)


class IMATCN(nn.Module):
    def __init__(
        self,
        input_features_num,
        input_len,
        output_len,
        tcn_OutputChannelList,
        tcn_KernelSize,
        tcn_Dropout,
        n_heads,
    ):
        super(IMATCN, self).__init__()
        embed_dim = tcn_OutputChannelList[-1]
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"IMATCN attention embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})."
            )

        self.tcnunit = TCNs(input_features_num, tcn_OutputChannelList, tcn_KernelSize, tcn_Dropout)
        self.attentionunit = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(embed_dim * input_len, output_len)

    def forward(self, input_seq):
        tcn_out = self.tcnunit(input_seq)
        tcn_out = tcn_out.permute(0, 2, 1)
        att_out, _ = self.attentionunit(tcn_out, tcn_out, tcn_out)
        att_out = att_out + tcn_out
        flatten_out = self.flatten(att_out)
        return self.linear(flatten_out)
