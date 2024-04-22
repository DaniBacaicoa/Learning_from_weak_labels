import torch
import pickle
import inspect
#from utils.losses import PartialLoss
import numpy as np

import torch.autograd as autograd


def train_model(model, trainloader, optimizer, loss_fn, num_epochs, return_model=False):
    model.train()

    train_losses = torch.zeros(num_epochs)
    train_accs = torch.zeros(num_epochs)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        for i, inputs, vl, trues, ind in enumerate(trainloader):
            #vl = vl.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args)>3:
                loss = loss_fn(outputs, vl,ind)
            else:
                loss = loss_fn(outputs, vl)
            loss.backward()
            optimizer.step()

            #Update batche's loss and acc
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(trues, dim=1)
            correct += torch.sum(preds == true)

        # loss and acc per epoch
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
    model.eval()
    correct = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct += torch.sum(preds == true)

    #Accuracy
    accuracy = correct.double() / len(testloader.dataset)

    # Print the accuracy if not in another function
    if sound:
        print('Evaluation Accuracy: {:.4f}'.format(accuracy))
    return accuracy


def train_and_evaluate(model, trainloader, testloader, optimizer, loss_fn, num_epochs, sound = 10):
    train_losses = torch.zeros(num_epochs)
    train_accs = torch.zeros(num_epochs)
    test_accs = torch.zeros(num_epochs)

    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0

        for inputs, vl, targets in trainloader:
            vl = vl.type(torch.LongTensor)
            inputs, vl, targets = inputs.to(device), vl.to(device), targets.to(device)
            
            
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args)>3:
                loss = loss_fn(outputs, vl, ind)
            else:
                loss = loss_fn(outputs, vl)
            loss.backward()
            optimizer.step()

            # Update batche's loss and acc
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct += torch.sum(preds == true)

        # loss and acc per epoch
        train_acc = correct.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        # Evaluate the model on the test set
        test_acc = evaluate_model(model, testloader, sound=False)
        test_accs[epoch] = test_acc

        if epoch % sound == sound-1:

            print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'
                  .format(epoch+1, num_epochs, train_loss, train_acc, test_acc))

    # Save the training and test results in a pickle file
    results = {'train_loss': train_losses, 'train_acc': train_accs, 'test_acc': test_accs}
    #with open('results.pkl', 'wb') as f:
    #    pickle.dump(results, f)

    return model, results


def train_and_evaluate_gradients(model, trainloader, testloader, optimizer, loss_fn, num_epochs, sound = 10):
    train_losses = torch.zeros(num_epochs)
    train_accs = torch.zeros(num_epochs)
    test_accs = torch.zeros(num_epochs)

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0

        for inputs, vl, targets in trainloader:
            vl = vl.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args)>3:
                loss = loss_fn(outputs, vl, ind)
            else:
                loss = loss_fn(outputs, vl)


            loss.backward(retain_graph=True)
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters()]))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print("Gradient is NaN or Inf!")
            else:
                print("Gradient Norm:", grad_norm.item())
            try:
                check_result = autograd.gradcheck(loss_fn, (outputs, vl), eps=1e-5)
                if not check_result:
                    print("Gradient check failed!")
                else:
                    print("Gradient check passed.")
            except Exception as e:
                print("Error during gradient check:", e)
            
            optimizer.step()

            # Update batche's loss and acc
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct += torch.sum(preds == true)

        # loss and acc per epoch
        train_acc = correct.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        train_losses[epoch] = train_loss
        train_accs[epoch] = train_acc

        # Evaluate the model on the test set
        test_acc = evaluate_model(model, testloader, sound=False)
        test_accs[epoch] = test_acc

        if epoch % sound == sound-1:

            print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'
                  .format(epoch+1, num_epochs, train_loss, train_acc, test_acc))

    # Save the training and test results in a pickle file
    results = {'train_loss': train_losses, 'train_acc': train_accs, 'test_acc': test_accs}
    #with open('results.pkl', 'wb') as f:
    #    pickle.dump(results, f)

    return model, results



def warm_up(model, trainloader, testloader, num_epochs):
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    optimizer = torch.optim.SGD(list(model.parameters()),lr = 1e-2, weight_decay = 1e-4, momentum = 0.9)
    loss_fn = PartialLoss(trainloader.dataset.tensors[1])

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0

        for inputs, wl, targets, ind in trainloader:

            outputs = model(inputs)
            loss = loss_fn(outputs,wl,ind)
            optimizer.zero_grad()
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
    results = {'train_loss': train_loss_list, 'train_acc': train_acc_list, 'test_acc': test_acc_list,}
    #with open('results.pkl', 'wb') as f:
    #    pickle.dump(results, f)

    # Return the trained model and the results
    return model, results





def ES_train_and_evaluate(model, trainloader, testloader, optimizer, loss_fn, num_epochs, patience=5):
    train_losses = []
    train_accs = []
    test_accs = []

    # variables for early stopping (this should be done on a validation set)
    # this is only for quick results
    #best_loss = np.inf
    best_acc = 0
    patience_counter = 0



    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        correct = 0

        for inputs, vl, targets, ind in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if len(inspect.getfullargspec(loss_fn.forward).args) > 3:
                loss = loss_fn(outputs, vl, ind)
            else:
                loss = loss_fn(outputs, vl)
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            _, true = torch.max(targets, dim=1)
            correct += torch.sum(preds == true)

        # Compute the training accuracy and loss
        train_acc = correct.double() / len(trainloader.dataset)
        train_loss = running_loss / len(trainloader.dataset)

        # loss and acc per epoch
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Evaluate the model on the test set
        test_acc = evaluate_model(model, testloader, sound=False)
        test_accs.append(test_acc)

        # Print the epoch results
        print('Epoch {}/{}: Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'
            .format(epoch+1, num_epochs, train_loss, train_acc, test_acc))

        # Early Stopping (on train, i know it should be on validation)
        if train_acc > best_acc:
            best_acc = train_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Train loss has not improved in {} epochs. Stopping early...'.format(patience))
                break
        '''        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Train loss has not improved in {} epochs. Stopping early...'.format(patience))
                break
        '''
        # Save the results
    results = {'train_loss': train_losses, 'train_acc': train_accs, 'test_acc': test_accs}

    return model, results
