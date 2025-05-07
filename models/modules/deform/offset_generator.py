from torch import nn


class OffsetGenerator(nn.Module):
    def __init__(self, cfg, in_ch, heads=None, offset_kernel=None, groups=1):
        super().__init__()
        self.cfg = cfg

        layer = nn.ModuleList()
        cfg_ = cfg["model"]["deform"]
        if heads is None:
            heads = cfg_["offset"]["heads"]
        self.offset_head = heads
        if offset_kernel is None:
            offset_kernel = cfg_["offset"]["kernel"][0]

        out_ch = 2 * offset_kernel * offset_kernel * heads * groups
        self.offset_layer = self._make_off_layers(in_ch, out_ch, kernel_size=3)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.init_weight_(m)
            if isinstance(m, (nn.ModuleList, nn.Sequential)):
                m.apply(self.init_weight_)

    def init_weight_(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x_traj):

        return self.offset_layer(x_traj)

    def _make_off_layers(self, in_channels, out_channels, kernel_size):
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=int(kernel_size // 2),
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.PReLU())

        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=int(kernel_size // 2),
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.PReLU())

        return nn.Sequential(*layers)
