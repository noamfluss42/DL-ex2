########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################
import os
import torch
from scipy.special import softmax
from scipy.special import expit

from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import pandas as pd
import matplotlib

from matplotlib import pyplot as plt

plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 14
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # TODO delete

batch_size = 32
output_size = 2
hidden_size = 128  # to experiment with

run_recurrent = False  # else run Token-wise MLP
use_RNN = True  # otherwise GRU
atten_size = 5  # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 2  # 10 is the original number
learning_rate = 0.0001
test_interval = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)
our_test_dataset = ld.get_our_test_data_set(batch_size)


# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


# Implements RNN Unit

class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # what else?
        self.in2output = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN"

    def forward(self, x, hidden_state):
        # Implementation of RNN cell
        a = torch.cat((x, hidden_state), 1)
        new_hidden = self.sigmoid(self.in2hidden(torch.cat((x, hidden_state), 1)))
        output = self.sigmoid(self.in2output(new_hidden))
        return output, new_hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


# Implements GRU Unit

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        # GRU Cell weights
        # self.something =
        # etc ...
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.zt_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.rt_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.ht_tilda_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"

    def forward(self, x, hidden_state):
        # Implementation of GRU cell
        zt = self.sigmoid(self.zt_linear(torch.cat((hidden_state, x), 1)))
        rt = self.sigmoid(self.rt_linear(torch.cat((hidden_state, x), 1)))
        ht_tilda = self.tanh(self.ht_tilda_linear(torch.cat((rt * hidden_state, x), 1)))
        new_hidden = (1 - zt) * hidden_state + zt * ht_tilda
        output = self.sigmoid(self.hidden_to_output(new_hidden))
        return output, new_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()
        self.sigmoid = torch.sigmoid
        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size, hidden_size)
        # additional layer(s)
        self.layer2 = MatMul(hidden_size, 512)

        self.layer3 = MatMul(512, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        # rest
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)
        self.sigmoid = torch.sigmoid
        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size, hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)
        self.layer2 = MatMul(hidden_size, output_size)
        # rest ...

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x)
        x = self.ReLU(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, atten_size:-atten_size, :]

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer
        N = 128  # TODO change to 100?
        query = self.W_q(x_nei)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)

        query_view = query.view(x.shape[0] * 100, query.shape[2], query.shape[3])
        keys_view = keys.view(x.shape[0] * 100, keys.shape[2], keys.shape[3])
        vals_view = vals.view(x.shape[0] * 100, vals.shape[2], vals.shape[3])

        d = torch.bmm(query_view, torch.transpose(keys_view, 1, 2)) / (N ** 0.5)
        alpha = self.softmax(d)
        weighted_values = torch.bmm(alpha, vals_view)
        output_view = torch.sum(weighted_values, 1)
        output_atten = output_view.view(x.shape[0], 100, output_view.shape[1])
        x = self.layer2(output_atten)
        x = self.sigmoid(x)
        return x, alpha


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, label, prediction):
    # implement #TODO smart coding

    for word_index in range(20):
        sub_scores = np.round([sbs1[word_index], sbs2[word_index]], 3)
        softmaxed_prediction = np.round(softmax(sub_scores), 3)
        print(
            f"word: '{rev_text[word_index]}', sub-scores:, {sub_scores}")  # Unclear softmaxed-prediction, {softmaxed_prediction}")
    print("final predicted label", prediction, "true label:", label)


# select model to use

def get_accuracy_score(output: torch.tensor, labels: torch.tensor):
    """
    :param output: the model output, tensor[samples,labels]
    :param labels: the original labels, tensor[samples,labels]
    :return: accuracy score
    """
    output_array = output.detach().numpy().argmax(axis=1)
    labels_array = labels.detach().numpy().argmax(axis=1)
    return np.where(output_array == labels_array)[0].size / labels_array.size


def plot_loss(train_losses, test_losses, model_name):
    plt.figure(figsize=(15, 8))
    plt.title(f"Train and Test Loss, model = {model_name}")
    plt.plot(np.arange(len(train_losses)), train_losses, '-', linewidth=2, label="Train Loss")
    plt.plot(np.arange(len(test_losses)), test_losses, '-', linewidth=2, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_acc(test_acc, model_name):
    plt.figure(figsize=(15, 8))
    plt.title(f"Test Accuracy, model = {model_name}")
    plt.plot(np.arange(len(test_acc)), test_acc, '-', linewidth=2, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def print_test_accuracy(model, test_dataset, run_recurrent, atten_size):
    acc = 0
    output = 0
    for labels, reviews, reviews_text in test_dataset:
        if run_recurrent:
            test_hidden_state = model.init_hidden(int(labels.shape[0])).to(device)
            for i in range(num_words):
                output, test_hidden_state = model(reviews[:, i, :], test_hidden_state)
        else:
            sub_score = []
            if atten_size > 0:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)

            output = torch.mean(sub_score, 1)
        acc += get_accuracy_score(output, labels)
        if not run_recurrent:
            nump_subs = sub_score.detach().numpy()
            labels = labels.detach().numpy()

    print(f"Final accuracy score = {acc / len(test_dataset)}")


def print_sub_score_words(model, our_test_dataset):
    for labels, reviews, reviews_text in our_test_dataset:
        sub_score = model(reviews)
        output = torch.mean(sub_score, 1)
        print("start print_review")
        nump_subs = sub_score.detach().numpy()
        labels_argmax = labels.detach().numpy().argmax(axis=1)
        nump_output_argmax = output.detach().numpy().argmax(axis=1)
        idx_sec = np.where(labels_argmax == nump_output_argmax)[0][0]
        sec_label = labels_argmax[idx_sec]
        idx_fail = np.where(labels_argmax != nump_output_argmax)[0][0]
        wrong_predict_label, real_label = nump_output_argmax[idx_fail], labels_argmax[idx_fail]
        print_review(reviews_text[idx_sec], nump_subs[idx_sec, :, 0], nump_subs[idx_sec, :, 1], sec_label, sec_label)
        print_review(reviews_text[idx_fail], nump_subs[idx_fail, :, 0], nump_subs[idx_fail, :, 1], real_label,
                     wrong_predict_label)


if __name__ == "__main__":
    if run_recurrent:
        if use_RNN:
            model = ExRNN(input_size, output_size, hidden_size)
        else:
            model = ExGRU(input_size, output_size, hidden_size)
    else:
        if atten_size > 0:
            model = ExLRestSelfAtten(input_size, output_size, hidden_size)
        else:
            model = ExMLP(input_size, output_size, hidden_size)

    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load(model.name() + ".pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    train_loss = 1.0
    test_loss = 1.0

    # training steps in which a test step is executed every test_interval
    train_loss_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(num_epochs):

        itr = 0  # iteration counter within each epoch

        for labels, reviews, reviews_text in train_dataset:  # getting training batches

            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
            else:
                test_iter = False

            # Recurrent nets (RNN/GRU)

            if run_recurrent:
                hidden_state = model.init_hidden(int(labels.shape[0])).to(device)
                output = 0
                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE

            else:

                # Token-wise networks (MLP / MLP + Atten.)

                if atten_size > 0:
                    # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else:
                    # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)

            # cross-entropy loss

            loss = criterion(output, labels)

            # optimize in training iterations

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                acc = get_accuracy_score(output, labels)
                test_acc_list.append(acc)
            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

            if test_iter:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Test Accuracy: {acc}"
                )

        # saving the model
        torch.save(model, model.name() + ".pth")

    plot_loss(train_loss_list, test_loss_list, model.name())
    plot_acc(test_acc_list, model.name())
    print_test_accuracy(model, test_dataset, run_recurrent, atten_size)

    print_sub_score_words(model, our_test_dataset)  # Question 2 - print sub scores MLP
    print('!')
