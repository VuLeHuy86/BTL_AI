"""Run the Q3 LinearSVC and Q4 character-CNN extensions."""

import argparse
import json
import time

import numpy as np

def run_q3_svm(project_dir):
    from sklearn.svm import LinearSVC

    data_path = project_dir / "data" / "mnist.npz"
    with np.load(data_path) as data:
        x_train = data["train_images"].reshape(-1, 28 * 28).astype(np.float32)
        y_train = data["train_labels"].astype(np.int64)
        # Match the official autograder split: odd-indexed original test rows.
        x_test = data["test_images"][1::2].reshape(-1, 28 * 28).astype(np.float32)
        y_test = data["test_labels"][1::2].astype(np.int64)

    started = time.perf_counter()
    model = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=42)
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started
    accuracy = float(np.mean(model.predict(x_test) == y_test))
    parameter_count = int(model.coef_.size + model.intercept_.size)
    result = {
        "experiment": "Q3 LinearSVC",
        "test_accuracy": accuracy,
        "train_seconds": train_seconds,
        "parameter_count": parameter_count,
        "train_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
    }
    print("RESULT " + json.dumps(result), flush=True)

def run_q4_cnn(project_dir):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = project_dir / "data" / "lang_id.npz"
    with np.load(data_path) as data:
        x_train = torch.tensor(data["train_x"], dtype=torch.long)
        y_train = torch.tensor(data["train_y"], dtype=torch.long)
        x_dev = torch.tensor(data["dev_x"], dtype=torch.long)
        y_dev = torch.tensor(data["dev_y"], dtype=torch.long)
        x_test = torch.tensor(data["test_x"], dtype=torch.long)
        y_test = torch.tensor(data["test_y"], dtype=torch.long)
        num_chars = int(len(data["chars"]))
        num_languages = int(len(data["language_names"]))

    class CharacterCNN(nn.Module):
        """Character embedding plus parallel 1D convolutions over word spelling."""
        def __init__(self):
            super().__init__()
            embedding_size = 32
            channels = 64
            self.embedding = nn.Embedding(num_chars + 1, embedding_size, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embedding_size, channels, kernel_size=k, padding="same")
                for k in (2, 3, 4)
            ])
            self.dropout = nn.Dropout(0.25)
            self.classifier = nn.Linear(channels * len(self.convs), num_languages)

        def forward(self, tokens):
            lengths = (tokens >= 0).sum(dim=1).clamp(min=1)
            embedded = self.embedding((tokens + 1).clamp(min=0)).transpose(1, 2)
            valid_positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths[:, None]
            pooled_features = []
            for conv in self.convs:
                features = torch.relu(conv(embedded))
                features = features.masked_fill(~valid_positions[:, None, :], torch.finfo(features.dtype).min)
                pooled_features.append(features.max(dim=2).values)
            joined = torch.cat(pooled_features, dim=1)
            return self.classifier(self.dropout(joined))

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    dev_loader = DataLoader(TensorDataset(x_dev, y_dev), batch_size=256)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256)
    model = CharacterCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_dev_accuracy = -1.0
    best_state = None
    stale_epochs = 0
    patience = 5
    max_epochs = 25
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.shape[0]
            total_count += labels.shape[0]

        model.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for tokens, labels in dev_loader:
                logits = model(tokens.to(device))
                correct += int((logits.argmax(dim=1).cpu() == labels).sum())
                count += labels.numel()
        dev_accuracy = correct / count
        average_loss = total_loss / total_count
        print(
            f"Q4 CNN epoch {epoch}/{max_epochs} | train loss: {average_loss:.4f} | "
            f"dev accuracy: {dev_accuracy:.2%}",
            flush=True,
        )
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Q4 CNN early stopping after {patience} stale epochs.", flush=True)
            break

    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for tokens, labels in test_loader:
            logits = model(tokens.to(device))
            correct += int((logits.argmax(dim=1).cpu() == labels).sum())
            count += labels.numel()
    accuracy = correct / count
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "experiment": "Q4 CharacterCNN1D",
        "test_accuracy": float(accuracy),
        "dev_accuracy_best": float(best_dev_accuracy),
        "train_seconds": train_seconds,
        "parameter_count": int(parameter_count),
        "train_examples": int(len(x_train)),
        "test_examples": int(len(x_test)),
        "device": str(device),
    }
    print("RESULT " + json.dumps(result), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", choices=("q3_svm", "q4_cnn"), required=True)
    parser.add_argument("--project-dir", type=str, default=".")
    args = parser.parse_args()
    project_dir = __import__("pathlib").Path(args.project_dir)
    if args.question == "q3_svm":
        run_q3_svm(project_dir)
    else:
        run_q4_cnn(project_dir)


if __name__ == "__main__":
    main()
