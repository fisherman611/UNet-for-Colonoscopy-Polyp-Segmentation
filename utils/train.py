from tqdm import tqdm
import torch 

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, save_path, wandb=None):
    """
    Train a model with a given dataset, optimizer, and loss function.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        criterion (torch.nn.Module): Loss function.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
        save_path (str): Path to save the best model checkpoint.
        wandb (wandb): Weights & Biases logging object (optional).

    Returns:
        None
    """
    # Move the model to the specified device
    model.to(device)

    # Initialize the best validation loss for checkpointing
    best_val_loss = float('inf')

    # Create a progress bar for tracking epochs
    epoch_bar = tqdm(total=num_epochs, desc='Training Progress')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).squeeze(1).long()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).squeeze(1).long()

                outputs = model(images)
                total_val_loss += criterion(outputs.float(), labels.long()).item()

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Log and update the progress bar
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.10f}")
        epoch_bar.set_postfix({'Train Loss': avg_train_loss, 'Val Loss': avg_val_loss})

        # Save the model if validation loss improves
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': total_val_loss,
            }
            torch.save(checkpoint, save_path)

        # Log metrics to Weights & Biases if provided
        if wandb:
            wandb.log({'Train Loss': avg_train_loss, 'Val Loss': avg_val_loss})

        epoch_bar.update(1)

    epoch_bar.close()