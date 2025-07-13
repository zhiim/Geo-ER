import os
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def train(
    model,
    dataset_train,
    dataset_val,
    criterion,
    device,
    save_path,
    batch_size=32,
    epochs=10,
    lr=3e-5,
):
    save_path = Path(save_path)

    model.to(device)

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size)

    opt = optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    num_steps = (len(train_loader) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * num_steps), num_training_steps=num_steps
    )

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        print("\n*** EPOCH:", epoch + 1, "***\n")

        for batch_idx, (source_text, pair_text, dist, label) in enumerate(
            train_loader
        ):
            opt.zero_grad()

            pred = model(source_text, pair_text, dist)
            loss = criterion(pred, label)
            loss.backward()
            opt.step()

            if device == "cuda":
                print(
                    "Step: {} / {}".format(
                        batch_idx + epoch * len(dataset_train),
                        epochs * len(dataset_train),
                    ),
                    "Loss: {}".format(loss.cpu().detach().numpy()),
                )
            else:
                print(
                    "Step: {} / {}".format(
                        batch_idx + epoch * len(dataset_train),
                        epochs * len(dataset_train),
                    ),
                    "Loss: {}".format(loss),
                )

            scheduler.step()

        print("\n*** Validation Epoch:", epoch + 1, "***\n")
        this_f1 = validate(
            model,
            val_loader,
            device,
        )

        if this_f1 > best_f1:
            best_f1 = this_f1
            torch.save(model.state_dict(), save_path / "model_best.pth")

        torch.save(
            model.state_dict(), save_path / "model_epoch_{}".format(epoch)
        )
        checkpoints = [
            chkpt
            for chkpt in os.listdir(save_path)
            if chkpt.endswith(".pth") and chkpt != "model_best.pth"
        ]
        if len(checkpoints) > 10:
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            os.remove(save_path / checkpoints[0])


def validate(model, val_loader, device):
    model.eval()

    acc = 0.0
    prec = 0.0
    recall = 0.0
    f1 = 0.0

    with torch.no_grad():
        for source_text, pair_text, dist, label in tqdm(val_loader):
            pred = model(source_text, pair_text, dist, training=False)

            acc += accuracy_score(label, torch.argmax(pred))
            prec += precision_score(label, torch.argmax(pred))
            recall += recall_score(label, torch.argmax(pred))
            f1 += f1_score(label, torch.argmax(pred))

    acc /= len(val_loader)
    prec /= len(val_loader)
    recall /= len(val_loader)
    f1 /= len(val_loader)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", recall)
    print("f1-Score:", f1)

    return f1
