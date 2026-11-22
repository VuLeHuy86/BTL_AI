from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        epoch = 0
        while True:
            epoch += 1
            made_mistake = False
            mistakes = 0
            for batch in dataloader:
                x = batch['x']
                label = batch['label']
                prediction = model.get_prediction(x)
                if prediction.item() != label.item():
                    model.w += label * x
                    made_mistake = True
                    mistakes += 1
            print(f"Q1 epoch {epoch} | mistakes: {mistakes}", flush=True)
            if not made_mistake:
                print("Q1 converged: no mistakes in a full pass.", flush=True)
                break


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    max_epochs = 2000
    target_loss = 0.015

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in dataloader:
            optimizer.zero_grad()
            predictions = model(batch['x'])
            loss = regression_loss(predictions, batch['label'])
            loss.backward()
            optimizer.step()
            count = batch['x'].shape[0]
            total_loss += loss.item() * count
            total_examples += count

        average_loss = total_loss / total_examples
        if epoch % 25 == 0 or average_loss <= target_loss:
            print(
                f"Q2 epoch {epoch}/{max_epochs} | train MSE: {average_loss:.6f}",
                flush=True,
            )
        if average_loss <= target_loss:
            print(f"Q2 early stop: MSE reached {target_loss}.", flush=True)
            break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    target_accuracy = 0.98
    max_epochs = 20

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in dataloader:
            optimizer.zero_grad()
            logits = model(batch['x'])
            loss = digitclassifier_loss(logits, batch['label'])
            loss.backward()
            optimizer.step()
            count = batch['x'].shape[0]
            total_loss += loss.item() * count
            total_examples += count

        model.eval()
        with no_grad():
            validation_accuracy = dataset.get_validation_accuracy()
        print(
            f"Q3 epoch {epoch}/{max_epochs} | train loss: "
            f"{total_loss / total_examples:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}",
            flush=True,
        )
        if validation_accuracy >= target_accuracy:
            print(f"Q3 early stop: validation accuracy reached {target_accuracy:.0%}.", flush=True)
            break


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    max_epochs = 20
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in dataloader:
            xs = movedim(batch['x'], 0, 1)
            optimizer.zero_grad()
            logits = model(xs)
            loss = languageid_loss(logits, batch['label'])
            loss.backward()
            optimizer.step()
            count = batch['x'].shape[0]
            total_loss += loss.item() * count
            total_examples += count

        # Train all scheduled epochs: this distribution's language dataset
        # names a method get_validation_accuracy, but it reads the test split.
        print(
            f"Q4 epoch {epoch}/{max_epochs} | train loss: "
            f"{total_loss / total_examples:.4f}",
            flush=True,
        )



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    max_epochs = 100
    target_accuracy = 0.84

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in dataloader:
            optimizer.zero_grad()
            logits = model(batch['x'])
            loss = digitconvolution_Loss(logits, batch['label'])
            loss.backward()
            optimizer.step()
            count = batch['x'].shape[0]
            total_loss += loss.item() * count
            total_examples += count

        model.eval()
        with no_grad():
            validation_accuracy = dataset.get_validation_accuracy()
        print(
            f"Q5 epoch {epoch}/{max_epochs} | train loss: "
            f"{total_loss / total_examples:.4f} | "
            f"validation accuracy: {validation_accuracy:.2%}",
            flush=True,
        )
        if validation_accuracy >= target_accuracy:
            print(f"Q5 early stop: validation accuracy reached {target_accuracy:.0%}.", flush=True)
            break
