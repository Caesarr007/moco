import moco.builder
import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, num_classes):
        super(mlp, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.model(x)


def main():

    feature_encoder = mlp

    model = moco.builder.MoCo(
        encoder = feature_encoder,
        dim = 128,
        K = 65536,
        m = 0.999,
        T = 0.07,
    )
    print(model)

    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, momentum=0.999, weight_decay=1e-4)



if __name__ == '__main__':
    main()

