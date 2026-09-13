# export_ts.py
import torch
import torch.nn as nn

class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

def main():
    net = DemoNet().eval()
    x = torch.randn(1,3,32,32)
    # 原生torch jit trace
    mod = torch.jit.trace(net, x)
    mod.save("demo_net.pt")
    print("✅ torchscript saved to demo_net.pt")

if __name__ == "__main__":
    main()
