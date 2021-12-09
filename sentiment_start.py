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
import wandb
from matplotlib import pyplot as plt

plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 14
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # TODO delete

reload_model = False

test_interval = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading sataset, use toy = True for obtaining a smaller dataset


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
    def __init__(self, layers_dim):
        super(ExRNN, self).__init__()
        self.input_size = layers_dim[0]
        self.hidden_size = layers_dim[1]
        self.output_size = layers_dim[-1]
        self.sigmoid = torch.sigmoid

        # RNN Cell weights
        self.in2hidden = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        # what else?
        self.in2output = nn.Linear(self.hidden_size, self.output_size)

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
    def __init__(self, layers_dim):
        super(ExGRU, self).__init__()
        self.input_size = layers_dim[0]
        self.hidden_size = layers_dim[1]
        self.output_size = layers_dim[-1]
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.zt_linear = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.rt_linear = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.ht_tilda_linear = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

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
    def __init__(self,
                 layers_dim,
                 layers_activations):
        super(ExMLP, self).__init__()
        self.input_size = layers_dim[0]
        self.output_size = layers_dim[-1]
        self.layers_activations = layers_activations
        self.sigmoid = torch.sigmoid
        self.layers = []
        # Token-wise MLP + Restricted Attention network implementation
        for i in range(len(layers_dim) - 1):
            self.layers.append(MatMul(layers_dim[i], layers_dim[i + 1]))
        self.layers = nn.ModuleList(self.layers)

    def name(self):
        return "MLP"

    def forward(self, x):
        # Token-wise MLP network implementation
        for layer_index in range(len(self.layers)):
            x = self.layers[layer_index](x)
            x = self.layers_activations[layer_index](x)
        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, atten_size,
                 layers_dim,
                 layers_activations, atten_layer_index):
        super(ExLRestSelfAtten, self).__init__()

        hidden_size = layers_dim[atten_layer_index]
        self.input_size = layers_dim[0]
        self.output_size = layers_dim[-1]
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.layers_activations = layers_activations
        self.atten_size = atten_size
        self.softmax = torch.nn.Softmax(2)
        self.layers = []
        # Token-wise MLP + Restricted Attention network implementation
        for i in range(len(layers_dim) - 1):
            self.layers.append(MatMul(layers_dim[i], layers_dim[i + 1]))

        self.W_q = MatMul(layers_dim[atten_layer_index], layers_dim[atten_layer_index], use_bias=False)
        self.W_k = MatMul(layers_dim[atten_layer_index], layers_dim[atten_layer_index], use_bias=False)
        self.W_v = MatMul(layers_dim[atten_layer_index], layers_dim[atten_layer_index], use_bias=False)

        self.atten_layer_index = atten_layer_index

        self.layers = nn.ModuleList(self.layers)

    def name(self):
        return "MLP_atten"

    def forward(self, x):
        # Token-wise MLP + Restricted Attention network implementation
        for layer_index in range(self.atten_layer_index):
            x = self.layers[layer_index](x)
            x = self.layers_activations[layer_index](x)

        # generating x in offsets between -atten_size and atten_size
        # with zero padding at the ends

        padded = pad(x, (0, 0, self.atten_size, self.atten_size, 0, 0))

        x_nei = []
        for k in range(-self.atten_size, self.atten_size + 1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei, 2)
        x_nei = x_nei[:, self.atten_size:-self.atten_size, :]

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer
        query = self.W_q(x_nei)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)

        query_view = query.view(x.shape[0] * 100, query.shape[2], query.shape[3])
        keys_view = keys.view(x.shape[0] * 100, keys.shape[2], keys.shape[3])
        vals_view = vals.view(x.shape[0] * 100, vals.shape[2], vals.shape[3])

        d = torch.bmm(query_view, torch.transpose(keys_view, 1, 2)) / self.sqrt_hidden_size
        alpha = self.softmax(d)
        weighted_values = torch.bmm(alpha, vals_view)
        output_view = torch.sum(weighted_values, 1)
        x = output_view.view(x.shape[0], 100, output_view.shape[1])

        for layer_index in range(self.atten_layer_index, len(self.layers)):
            x = self.layers[layer_index](x)
            x = self.layers_activations[layer_index](x)
        return x, alpha


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, label, prediction):
    # implement #TODO smart coding

    for word_index in range(len(rev_text)):
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
    output_array = output.cpu().detach().numpy().argmax(axis=1)
    labels_array = labels.cpu().detach().numpy().argmax(axis=1)
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


def get_test_accuracy(model, test_dataset, run_recurrent, atten_size, num_words):
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

    return acc / len(test_dataset)


def print_sub_score_words(model, our_test_dataset, atten_size):
    for labels, reviews, reviews_text in our_test_dataset:
        if atten_size > 0:
            sub_score, atten_weights = model(reviews)
        else:
            sub_score = model(reviews)
        output = torch.mean(sub_score, 1)
        print("start print_review")
        nump_subs = sub_score.cpu().detach().numpy()
        labels_argmax = labels.cpu().detach().numpy().argmax(axis=1)
        nump_output_argmax = output.cpu().detach().numpy().argmax(axis=1)
        print(f"labels_argmax = {labels_argmax}\noutput_argmax = {nump_output_argmax}")
        idx_first = 0
        first_predict, first_label = labels_argmax[idx_first], labels_argmax[idx_first]
        idx_second = 1
        second_predict, second_label = nump_output_argmax[idx_second], labels_argmax[idx_second]
        print_review(reviews_text[idx_first], nump_subs[idx_first, :, 0], nump_subs[idx_first, :, 1], first_label,
                     first_predict)
        print()
        print_review(reviews_text[idx_second], nump_subs[idx_second, :, 0], nump_subs[idx_second, :, 1], second_label,
                     second_predict)


def choose_model(run_recurrent, use_RNN, atten_size, layers_dim, layers_activations, atten_layer_index):
    if run_recurrent:
        if use_RNN:
            model = ExRNN(layers_dim)
        else:
            model = ExGRU(layers_dim)
    else:
        if atten_size > 0:
            model = ExLRestSelfAtten(atten_size,
                                     layers_dim,
                                     layers_activations, atten_layer_index)
        else:
            model = ExMLP(layers_dim, layers_activations)

    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load(model.name() + ".pth"))
    model.to(device)
    return model


def train(train_dataset, test_dataset, num_words, num_epochs, run_recurrent, atten_size, model, criterion, optimizer):
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
                output_train = output
                labels_train = labels

            # averaged losses
            if test_iter:
                test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                acc = get_accuracy_score(output, labels)
                acc_train = get_accuracy_score(output_train, labels_train)

                test_acc_list.append(acc)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Test Accuracy: {acc}"
                    f"train Accuracy: {acc_train}"
                )
                # wandb.log({"interval step batch test loss": test_loss})
                # wandb.log({"interval step batch train loss": train_loss})
                # wandb.log({"interval step test acc": acc})
                # wandb.log({"interval step train acc": acc_train})
            else:
                train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

        # wandb.log({"epoch step test acc": get_test_accuracy(model, test_dataset, run_recurrent, atten_size)})
        # saving the model
    torch.save(model, model.name() + ".pth")
    return train_loss_list, test_loss_list, test_acc_list


def run_by_architecture(train_dataset, test_dataset, num_words, our_test_dataset,
                        learning_rate=0.0001,
                        num_epochs=2,
                        run_recurrent=False,
                        use_RNN=True,
                        atten_size=5,
                        layers_dim=(100, 128, 2),
                        layers_activations=(torch.nn.ReLU(), torch.nn.ReLU(), torch.sigmoid),
                        atten_layer_index=1):
    model = choose_model(run_recurrent, use_RNN, atten_size, layers_dim, layers_activations, atten_layer_index)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_list, test_loss_list, test_acc_list = train(train_dataset, test_dataset, num_words, num_epochs,
                                                           run_recurrent, atten_size, model, criterion, optimizer)
    plot_loss(train_loss_list, test_loss_list, model.name())
    plot_acc(test_acc_list, model.name())
    print(f"Final accuracy score = {get_test_accuracy(model, test_dataset, run_recurrent, atten_size, num_words)}")
    if not run_recurrent:
        print_sub_score_words(model, our_test_dataset, atten_size)  # Question 2 - print sub scores MLP


def main():
    learning_rate = 0.0001
    num_epochs = 1  # 10 is the original number

    batch_size = 64

    run_recurrent = False  # else run Token-wise MLP
    use_RNN = True  # otherwise GRU
    atten_size = 5  # atten > 0 means using restricted self atten

    train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)
    our_test_dataset = ld.get_our_test_data_set(batch_size)

    layers_activations = (torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), torch.sigmoid)

    atten_layer_index = 1
    # for hidden_size in [64]:
    #     layers_dim = (100, hidden_size, 256, 2),
    #     config = {
    #         "learning_rate": learning_rate,
    #         "epochs": num_epochs,
    #         "batch_size": batch_size,
    #         "run_recurrent": run_recurrent,
    #         "use_RNN": use_RNN,
    #         "attention_after_layer": atten_layer_index,
    #         "layer1_size": layers_dim[0],
    #         "layer1_activation": "RELU",
    #         "layer2_size": layers_dim[1],
    #         "layer2_activation": "RELU",
    #         "layer3_size": layers_dim[2],
    #         "layer3_activation": "RELU",
    #         "layer4_size": layers_dim[3],
    #         "layer4_activation": "sigmoid",
    #     }
    #     wandb.init(project="DL-ex2_with_documentation_delete", entity="noam-fluss", config=config)
    run_by_architecture(train_dataset, test_dataset, num_words, our_test_dataset,
                        learning_rate,
                        num_epochs,
                        run_recurrent,
                        use_RNN,
                        atten_size,
                        layers_dim,
                        layers_activations,
                        atten_layer_index)


if __name__ == "__main__":
    # wandb.init(project="DL-ex2", entity="noam-fluss", config=config)
    main()
