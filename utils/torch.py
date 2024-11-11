'''
PyTorch tools.

Summary
-------
This module provides convenience tools for training and evaluating PyTorch models.
While most functions and classes have been written specifically for the OLX case study,
'BalancedSampler' and 'ClassifierTraining' were adapted from my own open-source repository.

'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import confusion_matrix


class BinarySet(Dataset):
    '''
    Binary dataset for a certain target class.

    Parameters
    ----------
    data : DataImport instance
        Data set under consideration.
    target : str or int
        Identifier of the target class.

    '''

    def __init__(self, data, target='clock', transform=None):
        self.data = data
        self.target = target
        self.transform = _create_transformer(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(self.data.images[idx])

        if self.data.labels is not None:
            label = 1 if self.data.labels[idx] == self.target else 0
            return image, label
        else:
            return image


def _create_transformer(transform):
    '''Create a transformer function..'''

    if transform is None: # just reshape
        def transformer(image):
            return image[None,...].transpose(0, 3, 1, 2)

    else: # torchvision transform
        def transformer(image):
            image = transform(image)
            return image

    return transformer


class SummedProbabilities(nn.Module):
    '''Model returning a sum of certain probabilities.'''

    def __init__(self, pretrained_model, target_ids):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.target_ids = target_ids

    def forward(self, x):
        probs = torch.softmax(self.pretrained_model(x), dim=1)
        summed_probs = torch.sum(probs[:,self.target_ids], dim=1)
        return summed_probs


def analyze_predictions(model, data_loader, threshold=None, k=None, target_ids=None):
    '''Analyze the model predictions w.r.t. a data loader.'''

    y_pred, y_true = predict_loader(model, data_loader, return_true=True)

    # thresholded probabilities
    if threshold is not None and k is None and target_ids is None:
        y_pred = (y_pred >= threshold).type(y_true.dtype)

    # top-k predictions
    elif k is not None and target_ids is not None and threshold is None:
        top_prob, top_class = torch.topk(y_pred, k=k, dim=1)

        y_pred = torch.tensor([torch.any(torch.tensor([x in class_ids for x in target_ids]))
                               for class_ids in top_class], dtype=y_true.dtype)

    # confusion matrix
    confusion = confusion_matrix(y_true.data.numpy(), y_pred.data.numpy())

    tn, fp, fn, tp = confusion.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    summary = {
        'confusion': confusion,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

    return summary


@torch.no_grad()
def predict_loader(model, data_loader, return_true=True):
    '''Compute model predictions over a data loader.'''

    model.eval()

    pred_list = []
    true_list = []

    for X_batch, y_batch in data_loader:
        true_list.append(y_batch)
        pred_list.append(model(X_batch)) # TODO: Enable GPU support

    y_pred = torch.cat(pred_list, dim=0)
    y_true = torch.cat(true_list, dim=0)

    if return_true:
        return y_pred, y_true
    else:
        return y_pred


class BalancedSampler(Sampler):
    '''
    Balanced sampling of imbalanced datasets.

    Summary
    -------
    In order to deal with an imbalanced classification dataset,
    an appropriate over/undersampling scheme is implemented.
    Here, samples are taken with replacement from the set, such that
    all classes are equally likely to occur in the training mini-batches.
    This might be especially helpful in combination with data augmentation.
    Different weights for samples in the empirical loss would be an alternative.

    Parameters
    ----------
    data_set : PyTorch dataset
        Imbalanced dataset to be over/undersampled.
    no_samples : int or None
        Number of samples to draw in one epoch.
    indices : array_like or None
        Subset of indices that are sampled.

    '''

    def __init__(self, dataset, no_samples=None, indices=None):

        self.indices = list(range(len(dataset)))

        if no_samples is None:
            self.no_samples = len(dataset) if indices is None else len(indices)
        else:
            self.no_samples = no_samples

        # class occurrence counts
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        labels_list = []

        for image, label in data_loader:
            labels_list.append(label)

        labels_tensor = torch.cat(labels_list, dim=0)

        unique_labels, counts = torch.unique(labels_tensor, return_counts=True)

        # unnormalized probabilities
        weights_for_class = 1. / counts.float()

        weights_for_index = torch.tensor(
            [weights_for_class[labels_tensor[idx]] for idx in self.indices]
        )

        # zero indices
        if indices is not None:
            zero_ids = np.setdiff1d(self.indices, indices).tolist()

            weights_for_index[zero_ids] = torch.tensor(0.)

        # balanced sampling distribution
        self.categorical = torch.distributions.Categorical(probs=weights_for_index)

    def __iter__(self):
        return (idx for idx in self.categorical.sample((self.no_samples,)))

    def __len__(self):
        return self.no_samples


class ClassifierTraining:
    '''
    Training classifier models.

    Summary
    -------
    This class facilitates the training and testing of PyTorch models.
    It features methods performing a whole training loop and a single epoch.
    Testing on a full data loader is also implemented as a method.
    The most common PyTorch loss functions nn.BCEWithLogitsLoss,
    nn.CrossEntropyLoss and nn.NLLLoss are supported at the moment.
    The implementation aims at being device-agnostic,
    i.e. a GPU is used whenever available, the CPU is used otherwise.

    The class supports both binary and multi-class classification problems.
    For the binary case, it is assumed that the model does not involve a final (log)-sigmoid,
    i.e. the sigmoid is applied to the model output for testing purposes only.
    For the multiclass classification, the model may or may not perform a (log)-softmax operation.
    During testing, only the class with the highest response is compared to ground truth,
    the results of which are not altered under the normalizing softmax function.
    Note that, whenever desired, the logit scores can be easily casted as probabilities.

    Parameters
    ----------
    model : PyTorch module
        Model to be trained.
    criterion : PyTorch loss function
        Loss function criterion.
    optimizer : PyTorch optimizer
        Optimization routine.
    train_loader : PyTorch data loader
        Loader that generates the training data.
    val_loader : PyTorch data loader
        Loader that generates the validation data.
    device : PyTorch device
        Device the computations are performed on.

    '''

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader=None,
        device=None
    ):

        # arguments
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # device
        if device is None:
            # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.device = torch.device('cpu') # TODO: Enable GPU support
        else:
            self.device = device

        self.model = self.model.to(self.device)

    def __call__(self, X):
        '''Call model.'''
        y = self.model(X)
        return y

    def predict(self, X):
        '''Predict outputs.'''
        y = self(X)
        return y

    def fit(
        self,
        no_epochs,
        log_interval=100,
        threshold=0.5,
        initial_test=True
    ):
        '''Perform a number of training epochs.'''

        self.epoch = 0

        train_losses = []
        val_losses = []
        val_accs = []

        # initial test
        if initial_test:

            train_loss, train_acc = self.test(self.train_loader, threshold=threshold)
            val_loss, val_acc = self.test(self.val_loader, threshold=threshold)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print('Started training: {}, avg. val. loss: {:.4f}, val. acc.: {:.4f}' \
                  .format(self.epoch, val_loss, val_acc))

        # training loop
        for epoch_idx in range(no_epochs):

            self.epoch += 1

            train_loss = self.train_epoch(log_interval, threshold=threshold)

            train_losses.append(train_loss)

            if self.val_loader is not None:

                val_loss, val_acc = self.test(threshold=threshold)

                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print('Finished epoch: {}, avg. val. loss: {:.4f}, val. acc.: {:.4f}' \
                      .format(self.epoch, val_loss, val_acc))

        history = {
            'no_epochs': no_epochs,
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_acc': val_accs
        }

        return history

    def train_epoch(self, log_interval=100, threshold=0.5):
        '''Perform a single training epoch.'''

        self.model.train()

        batch_losses = []

        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):

            # device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # forward
            y_pred = self.model(X_batch)

            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                y_batch = y_batch.view(*y_pred.shape).float()
            elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                y_batch = y_batch.view(-1)

            loss = self.criterion(y_pred, y_batch)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # analysis
            batch_loss = loss.data.item()

            batch_losses.append(batch_loss)

            if len(batch_losses) < 3:
                running_loss = batch_loss
            else:
                running_loss = _moving_average(batch_losses, window=3, mode='last')

            no_total = X_batch.shape[0]

            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                is_correct = (torch.sigmoid(y_pred) >= threshold).squeeze().int() == y_batch.squeeze().int()
            elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                is_correct = torch.max(y_pred, dim=1)[1].squeeze().int() == y_batch.squeeze().int()

            no_correct = torch.sum(is_correct).item()

            batch_acc = no_correct / no_total

            if log_interval is not None:
                if (batch_idx+1) % log_interval == 0 or (batch_idx+1) == len(self.train_loader):
                    print('Epoch: {} ({}/{}), batch loss: {:.4f}, batch acc.: {:.4f}' \
                          .format(self.epoch+1, batch_idx+1, len(self.train_loader), batch_loss, batch_acc))

        return running_loss

    @torch.no_grad()
    def test(self, test_loader=None, no_epochs=1, threshold=0.5):
        '''Compute average test loss and accuracy.'''

        if test_loader is None:
            test_loader = self.val_loader

        self.model.eval()

        no_total = 0
        no_correct = 0
        test_loss = 0.

        for epoch_idx in range(no_epochs):

            for X_batch, y_batch in test_loader:

                # device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # forward
                y_pred = self.model(X_batch)

                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    y_batch = y_batch.view(*y_pred.shape).float()
                elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    y_batch = y_batch.view(-1)

                loss = self.criterion(y_pred, y_batch)

                # analysis
                test_loss += loss.data.item()

                no_total += X_batch.shape[0]

                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    is_correct = (torch.sigmoid(y_pred) >= threshold).squeeze().int() == y_batch.squeeze().int()
                elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    is_correct = torch.max(y_pred, dim=1)[1].squeeze().int() == y_batch.squeeze().int()

                no_correct += torch.sum(is_correct).item()

        test_acc = no_correct / no_total

        if self.criterion.reduction == 'sum': # averaging over all data
            test_loss /= len(test_loader.dataset)
        elif self.criterion.reduction == 'mean': # averaging over batches
            test_loss /= len(test_loader)

        return test_loss, test_acc


def _moving_average(x, window=3, mode='full'):
    '''
    Calculate the moving average over an array.

    Summary
    -------
    This function computes the running mean of an array.
    Padding is performed for the 'left' side, not for the 'right'.

    Parameters
    ----------
    x : array
        Input array.
    window : int
        Window size.
    mode : {'full', 'last'}
        Determines whether the full rolling mean history
        or only its last element is returned.

    Returns
    -------
    running_mean : float
        Rolling mean.

    '''

    x = np.array(x)

    if mode == 'full':

        x_padded = np.pad(x, (window-1, 0), mode='constant', constant_values=x[0])
        running_mean = np.convolve(x_padded, np.ones((window,))/window, mode='valid')

    elif mode == 'last':

        if x.size >= window:
            running_mean = np.convolve(x[-window:], np.ones((window,))/window, mode='valid')[0]
        else:
            x_padded = np.pad(x, (window-x.size, 0), mode='constant', constant_values=x[0])
            running_mean = np.convolve(x_padded, np.ones((window,))/window, mode='valid')[0]

    return running_mean

