import numpy as np
import torch as t
from sklearn.metrics import f1_score


class Trainer:

    def __init__(self, model, loss_function, optimizer, train_data_loader, test_data_loader, cuda=True):
        self.__model = model

        self.__loss_function = loss_function

        self.__optimizer = optimizer

        self.__train_data_loader = train_data_loader

        self.__test_data_loader = test_data_loader

        self.__cuda = cuda

        if cuda:
            self.__model = model.cuda()

            self.__loss_function = loss_function.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self.__model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self.__cuda else None)

        self.__model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self.__model.cpu()

        m.eval()

        x = t.randn(1, 512, requires_grad=True)

        y = self.__model(x)

        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        self.__optimizer.zero_grad()

        outputs = self.__model(x)

        outputs = outputs.view(outputs.size(0), -1)

        loss = self.__loss_function(outputs, y)

        loss.backward()

        self.__optimizer.step()

        return loss

    def val_test_step(self, x, y):

        with t.no_grad():
            outputs = self.__model(x)

            outputs = outputs.view(outputs.size(0), -1)

            loss = self.__loss_function(outputs, y)

        return loss, outputs

    def train_epoch(self):

        running_loss = 0.0

        for sample in self.__train_data_loader:

            features, labels = sample

            if self.__cuda:
                features = features.cuda()

                labels = labels.cuda()

            loss = self.train_step(features, labels)

            running_loss += loss.item()

        avg_loss = running_loss / len(self.__train_data_loader)

        print("Training dataset: Achieved average loss of: %.2f" % avg_loss)

        return avg_loss

    def val_test(self):

        running_loss = 0.0

        with t.no_grad():

            targets = list()

            predictions = list()

            for sample in self.__test_data_loader:

                features, labels = sample

                if self.__cuda:
                    features = features.cuda()

                    labels = labels.cuda()

                loss, output = self.val_test_step(features, labels)

                running_loss += loss.item()

                output = output.cpu().numpy()

                output = list(np.where(output >= 0.5, 1, 0))

                output = [float(o) for o in output]

                labels = list(labels.cpu().numpy().astype(int))

                labels = [float(label) for label in labels]

                targets.extend(labels)

                predictions.extend(output)

        avg_loss = running_loss / len(self.__test_data_loader)

        f1 = f1_score(targets, predictions, average='macro')

        print("Validation dataset: Achieved mean F1 of: %.2f" % f1)

        print("Validation dataset: Achieved mean loss of: %.2f" % avg_loss)

        return avg_loss

    def fit(self, epochs):

        current_epoch = 0

        train_losses = list()

        validation_losses = list()

        while current_epoch < epochs:
            training_loss = self.train_epoch()

            validation_loss = self.val_test()

            train_losses.append(training_loss)

            validation_losses.append(validation_loss)

            self.save_checkpoint(current_epoch)

            current_epoch += 1

        return train_losses, validation_losses
