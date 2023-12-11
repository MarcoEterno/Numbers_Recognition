# Assume `model` is your model
model = model
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# For early stopping
n_epochs_stop = 5
min_val_loss = float('inf')
epochs_no_improve = 0

# For model checkpointing
model_path = 'model.pt'

for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    val_loss = 0.0

    # Training loop
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation loop
    # Assume you have a validation dataloader `val_dataloader`
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # If the validation loss is at a minimum
    if val_loss < min_val_loss:
        # Save the model
        torch.save(model, model_path)
        epochs_no_improve = 0
        min_val_loss = val_loss

    else:
        epochs_no_improve += 1
        # Check early stopping condition
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            # Load in the best model
            model = torch.load(model_path)
            break
    else:
        continue

    print(f'Epoch {epoch+1}, Train Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss/len(val_dataloader)}')

print('Finished Training')