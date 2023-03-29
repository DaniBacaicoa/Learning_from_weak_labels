import torch
import pickle
import inspect

def train_model(model,trainloader, optimizer, loss_fn, num_epochs, return_model=False):
    # Set the model to training mode
    model.train()

    # Initialize the loss and accuracy tensors
    train_losses = torch.zeros(num_epochs)
    train_accs = torch.zeros(num_epochs)

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        # Initialize the running loss and correct predictions
        running_loss = 0.0
        correct = 0

        # Iterate over the training batches
        for inputs, vl, trues, ind in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args)>3:
                loss = loss_fn(outputs, vl,ind)
            else:
                loss = loss_fn(outputs, vl)
            loss.backward()
            optimizer.step()

            # Update the running loss and correct predictions
            running_loss += loss.item()
            _, preds = torch.max(outputs,dim=1)
            _, true = torch.max(trues,dim=1)
            correct += torch.sum(preds == true)

        # Compute the average loss and accuracy for this epoch
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct.double() / len(trainloader.dataset)

        # Store and print loss and accuracy for every epoch
        train_losses[epoch] = epoch_loss
        train_accs[epoch] = epoch_acc
        print('Epoch {}/{} - Loss: {:.4f} - Accuracy: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))
    if return_model:
        return train_losses, train_accs, model
    else:
        return train_losses, train_accs

def evaluate_model(model, testloader, sound = True):
    # Set the model to evaluation mode
    model.eval()
    correct = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)

            # Compute the predictions
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct += torch.sum(preds == true)

    # Compute the accuracy
    accuracy = correct.double() / len(testloader.dataset)

    # Print the accuracy
    if sound:
        print('Evaluation Accuracy: {:.4f}'.format(accuracy))

    # Return the accuracy
    return accuracy



def train_and_evaluate(model, trainloader, testloader, optimizer, loss_fn, num_epochs):

    # Create a list to store the training loss, training accuracy, and test accuracy
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Iterate over the epochs
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0

        # Iterate over the training batches
        for inputs, vl, targets, ind in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args)>3:
                loss = loss_fn(outputs, vl, ind)
            else:
                loss = loss_fn(outputs, vl)
            #loss = loss_fn(outputs, vl)
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)

            # Update the correct predictions
            correct += torch.sum(preds == true)

        # Compute the training accuracy and loss
        train_acc = correct.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        # Append the training accuracy and loss to the lists
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Evaluate the model on the test set
        test_acc = evaluate_model(model, testloader, sound=False)

        # Append the test accuracy to the list
        test_acc_list.append(test_acc)

        # Print the epoch results
        print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, test_acc))

    # Save the training and test results in a pickle file
    results = {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'test_acc': test_acc_list}
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Return the trained model and the results
    return model, results



def warm_up(model, trainloader, testloader, optimizer, loss_fn, num_epochs):

    # Create a list to store the training loss, training accuracy, and test accuracy
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # Iterate over the epochs
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0

        # Iterate over the training batches
        for inputs, vl, targets, ind in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args)>3:
                loss = loss_fn(outputs, vl, ind)
            else:
                loss = loss_fn(outputs, vl)
            #loss = loss_fn(outputs, vl)
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)

            # Update the correct predictions
            correct += torch.sum(preds == true)

        # Compute the training accuracy and loss
        train_acc = correct.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        # Append the training accuracy and loss to the lists
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Evaluate the model on the test set
        test_acc = evaluate_model(model, testloader, sound=False)

        # Append the test accuracy to the list
        test_acc_list.append(test_acc)

        # Print the epoch results
        print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, test_acc))

    # Save the training and test results in a pickle file
    results = {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'test_acc': test_acc_list}
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Return the trained model and the results
    return model, results