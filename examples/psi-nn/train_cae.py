import torch
from torch.utils.data import DataLoader

from linkefl.common.const import Const
from linkefl.dataio import MediaDataset
from linkefl.modelzoo import AutoEncoder


def loss_fn(original_y, encoder_y, decoder_y, gamma1, gamma2):
    ce = torch.nn.CrossEntropyLoss()
    entropy = -1 * torch.sum(encoder_y * torch.log2(encoder_y))
    loss = ce(original_y, decoder_y)
    loss1 = ce(original_y, encoder_y)
    loss2 = entropy
    return loss - gamma1 * loss1 - gamma2 * loss2


if __name__ == "__main__":
    torch.cuda.empty_cache()
    dataset_name = "cifar10"
    dataset_dir = "data"
    num_classes = 10
    epochs = 50
    batch_size = 64
    learning_rate = 0.1
    gamma1 = 0.1
    gamma2 = 0.1
    device = "cuda:0"
    active_trainset = MediaDataset(
        role=Const.ACTIVE_NAME,
        dataset_name=dataset_name,
        root=dataset_dir,
        train=True,
        download=True,
    )
    dataloader = DataLoader(
        active_trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = AutoEncoder(num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

    # training
    model.train()
    for epoch in range(epochs):
        for batch_idx, (_, y) in enumerate(dataloader):
            print(f"batch: {batch_idx}")
            y_onehot = torch.zeros(y.size(0), num_classes)
            y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
            y_onehot = y_onehot.to(device)
            decoded_y, confused_y = model(y_onehot)
            loss = loss_fn(y_onehot, confused_y, decoded_y, gamma1, gamma2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * len(y)
                print(f"loss: {loss:>7f} [{current:>5d}/{len(active_trainset):>5d}]")

    print("done")
