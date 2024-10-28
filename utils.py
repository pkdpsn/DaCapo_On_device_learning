import torch
import time
device = 'cpu'## "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device=device):
    train_loss, train_acc = 0, 0
    model.train()

    total_forward_time = 0.0
    total_backward_time = 0.0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass timing
        start_forward = time.time()
        y_pred = model(X)
        forward_time = time.time() - start_forward
        total_forward_time += forward_time

        # Loss computation
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Accuracy calculation
        acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        train_acc += acc

        # Backward pass timing
        start_backward = time.time()
        optimizer.zero_grad()
        loss.backward()
        backward_time = time.time() - start_backward
        total_backward_time += backward_time

        # Step
        optimizer.step()

    # Calculate average loss and accuracy
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Total Forward Time: {total_forward_time:.3f} seconds")
    print(f"Total Backward Time: {total_backward_time:.3f} seconds")
    print(f"Train Loss: {train_loss:.2f} | Train Accuracy: {train_acc:.2f}")

    return train_loss, train_acc
def test_step(model, data_loader,loss_fn,optimizer,accuracy_fn,device=device):
    test_time_stamp = time.time()
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X,y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss
            acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
            test_acc += acc
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        test_time_stamp = time.time() - test_time_stamp
        print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}% | Test Time: {test_time_stamp:.2f}s")
