import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def train(model,
          train_dataloader,
          test_dataloader,
          optimizer,
          loss_fn,
          accuracy_fn,
          epochs,
          device):
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch+1}\n---------")
        train_step(model=model,
                   data_loader=train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device)
        test_step(model=model,
                  data_loader=test_dataloader,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn,
                  device=device)

def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               return_preds: bool = False):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        return_preds (bool, optional): If True, also returns predicted and true labels. Defaults to False.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    # Initialize accumulators for loss and accuracy
    loss, acc = 0, 0

    # Initialize lists to store predictions and true labels if needed
    all_preds, all_labels = [], []

    # Set model to evaluation mode
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                               y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

            # Optionally store predictions and labels for further analysis
            if return_preds:
                all_preds.append(y_pred.argmax(dim=1).cpu())
                all_labels.append(y.cpu())

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    # Build the results dictionary
    results = {
        "model_name": model.__class__.__name__, # only works when model was created with a class
        "model_loss": loss.item(),
        "model_acc": acc
    }

    # Optionally include predictions and labels
    if return_preds:
        results["y_preds"] = torch.cat(all_preds)
        results["y_trues"] = torch.cat(all_labels)

    return results

    
def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch

def plot_confusion_matrix_from_results(results: dict, class_names: list):
    """
    Plots a confusion matrix from eval_model results dictionary.

    Args:
        results (dict): Dictionary returned from eval_model with keys "y_preds" and "y_trues".
        class_names (list): List of class names corresponding to label indices.

    Returns:
        Matplotlib figure and axis with plotted confusion matrix.
    """
    # 1. Extract predictions and true labels
    y_preds = results["y_preds"]
    y_trues = results["y_trues"]

    # 2. Setup confusion matrix instance and compute matrix
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_preds, target=y_trues)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib prefers NumPy
        class_names=class_names,          # label rows/columns
        figsize=(10, 7)
    )
    plt.title(f"Confusion Matrix: {results['model_name']}")
    plt.show()

    return fig, ax