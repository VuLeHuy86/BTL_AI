"""Train the optional character-level GPT on the Shakespeare corpus."""

import torch
from torch.utils.data import Dataset, DataLoader

from gpt_model import Character_GPT


class CharDataset(Dataset):
    def __init__(self, data, block_size=10):
        if len(data) <= block_size:
            raise ValueError("input.txt must contain more characters than block_size.")

        self.block_size = block_size
        chars = sorted(set(data))
        self.index2char = {i: ch for i, ch in enumerate(chars)}
        self.char2index = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        encoded = [self.char2index[ch] for ch in chunk]
        return (
            torch.tensor(encoded[:-1], dtype=torch.long),
            torch.tensor(encoded[1:], dtype=torch.long),
        )


def train_single_iteration(model, data_iter, train_loader, optimizer, device):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y = next(data_iter)

    x = x.to(device)
    y = y.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = model.get_loss(x, y)
    loss.backward()
    optimizer.step()
    return data_iter, loss.item()


def main():
    learning_rate = 0.0004
    num_iterations = 10000
    batch_size = 128
    checkpoint_interval = 100
    layer_size = 100
    n_layer = 6
    block_size = 10

    try:
        with open('input.txt', 'r', encoding='utf-8') as corpus_file:
            text = corpus_file.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Place the Shakespeare corpus in input.txt in the project folder."
        ) from exc

    dataset = CharDataset(text, block_size=block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        num_workers=0,
    )
    data_iter = iter(train_loader)

    model = Character_GPT(
        dataset.block_size,
        n_embd=layer_size,
        n_layer=n_layer,
        vocab_size=dataset.vocab_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    prompt = text[:min(block_size, len(text))]
    prompt_ids = torch.tensor(
        [[dataset.char2index[ch] for ch in prompt]],
        dtype=torch.long,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Corpus characters: {len(text)}")
    print(f"Vocabulary size: {dataset.vocab_size}")

    running_loss = 0.0
    model.train()
    for iteration in range(1, num_iterations + 1):
        data_iter, loss = train_single_iteration(
            model, data_iter, train_loader, optimizer, device
        )
        running_loss += loss

        if iteration % checkpoint_interval == 0:
            average_loss = running_loss / checkpoint_interval
            running_loss = 0.0
            print(
                f"Iteration {iteration}/{num_iterations} | train loss: {average_loss:.4f}",
                flush=True,
            )

            model.eval()
            with torch.no_grad():
                generated = model.generate(prompt_ids, 200)[0].tolist()
            sample = ''.join(dataset.index2char[index] for index in generated)
            print(f"Prompt: {prompt}")
            print(f"Generated text: {sample}\n", flush=True)
            model.train()


if __name__ == '__main__':
    main()
