from tqdm.auto import tqdm
import torch

# function to train a model in a training loop
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
    ):
    # model in training mode
    model.train()
    # putting model to device
    model.to(device)
    train_loss = 0
    train_acc = 0
    i = 0
    for batch, data in enumerate(dataloader):
        i += 1
        X, Y = data
        # Putting data into device
        X, Y = X.to(device), Y.to(device)
        # forward pass
        Y_pred = model(X)
        # loss
        loss = loss_fn(Y_pred, Y)
        train_loss += loss
        # accuracy
        acc = (Y == torch.argmax(torch.softmax(Y_pred, dim = 1), dim = 1)).sum().item()*100/len(Y)
        train_acc += acc
        # zero grad
        optimizer.zero_grad()
        # back propagation
        loss.backward()
        # optimizer step
        optimizer.step()

    train_loss /= i
    train_acc /= i
    return train_loss, train_acc

# function to test or evaluate model in each step on test data
def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: str):
    # model in inference mode
    model.eval()
    with torch.inference_mode():
        # model on device
        model.to(device)
        test_loss = 0
        test_acc = 0
        i = 0
        for batch, data in enumerate(dataloader):
            i += 1
            X, Y = data
            # putting data on device
            X, Y = X.to(device), Y.to(device)
            # forward pass
            Y_pred = model(X)
            # loss
            loss = loss_fn(Y_pred, Y)
            test_loss += loss
            # acc
            acc = (Y == torch.argmax(torch.softmax(Y_pred, dim = 1), dim = 1)).sum().item()*100/len(Y)
            test_acc += acc

        test_loss /= i
        test_acc /= i
        return test_loss, test_acc
    
def train(epochs: int,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: str
          ):
    epoch_count = []
    train_loss_count = []
    train_acc_count = []
    test_loss_count = []
    test_acc_count = []
    i = 0
    for epoch in tqdm(range(epochs)):
        i += 1
        train_loss, train_acc = train_step(model,
                train_dataloader,
                loss_fn,
                optimizer,
                device)
        test_loss, test_acc = test_step(model,
                test_dataloader,
                loss_fn,
                device)
        epoch_count.append(i)
        train_loss_count.append(train_loss.item())
        train_acc_count.append(train_acc)
        test_loss_count.append(test_loss.item())
        test_acc_count.append(test_acc)
        print(f"Epoch : {epoch} | Train Loss : {train_loss:.4f} | Train Accuracy : {train_acc:.2f} | Test Loss : {test_loss:.4f} | Test Accuracy : {test_acc:.2f}")
    return epoch_count, train_loss_count, train_acc_count, test_loss_count, test_acc_count

