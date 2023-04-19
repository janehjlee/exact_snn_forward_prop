import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
import snntorch.functional as SF

import scipy.sparse as sp
import scipy.sparse.linalg
import time


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, help="train and eval batch size")
parser.add_argument('--n_epochs', type=int, help="number of training epochs")
parser.add_argument('--spike_grad', help="surrogate gradient options are atan, fast sigmoid, sigmoid, sre, ste, tri")

args = parser.parse_args()


# Define Network
class Net(nn.Module):
    def __init__(self,
                 in_dim,
                 layer_size,
                 out_size,
                 num_steps,
                 alpha,
                 beta,
                 threshold,
                 layer_means=[1.5, 1.5],
                 layer_stds=[0.8, 0.8],
                 bias=[0.0, 0.0],
                 fc1_weight=None,
                 fc2_weight=None,
                 spike_grad=snn.surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        
        self.num_steps = num_steps

        # initialize layers
        self.fc1 = nn.Linear(in_dim, layer_size)
        if fc1_weight is not None:
            self.fc1.weight = torch.nn.Parameter(torch.Tensor(fc1_weight))
        else:
            nn.init.normal_(self.fc1.weight, layer_means[0], layer_stds[0])
        nn.init.constant_(self.fc1.bias, bias[0])
        
        self.lif1 = snn.Synaptic(alpha=alpha, 
                                 beta=beta, 
                                 threshold=threshold, 
                                 reset_mechanism='subtract', 
                                 spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(layer_size, out_size)
        if fc2_weight is not None:
            self.fc2.weight = torch.nn.Parameter(torch.Tensor(fc2_weight))
        else:
            nn.init.normal_(self.fc2.weight, layer_means[1], layer_stds[1])
        nn.init.constant_(self.fc2.bias, bias[1])
        
        self.lif2 = snn.Synaptic(alpha=alpha, 
                                 beta=beta, 
                                 threshold=threshold, 
                                 reset_mechanism='subtract',
                                 spike_grad=spike_grad)
        

    def forward(self, x):
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        
        spk1_rec=[]
        mem1_rec=[]

        spk2_rec = []  # Record the output trace of spikes
        mem2_rec = []  # Record the output trace of membrane potential

        for step in range(self.num_steps):

            cur1 = self.fc1(x[step])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            
            cur2 = self.fc2(spk1)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return [torch.stack(spk1_rec, dim=0), 
                torch.stack(mem1_rec, dim=0),
                torch.stack(spk2_rec, dim=0), 
                torch.stack(mem2_rec, dim=0)]
    
    
    
#from torch.utils.data.dataset import Dataset
        
class YinYangDataset(Dataset):
    def __init__(self, r_small=0.1, r_big=0.5, size=1000, seed=42, transform=None, **kwargs):
        super(YinYangDataset, self).__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(seed)
        self.transform = transform
        self.r_small = r_small
        self.r_big = r_big
        self.__vals = []
        self.__cs = []
        self.class_names = ['yin', 'yang', 'dot']
        for i in range(size):
            # keep num of class instances balanced by using rejection sampling
            # choose class for this sample
            goal_class = self.rng.randint(3)
            x, y, c = self.get_sample(goal=goal_class)
            # add mirrod axis values
            x_flipped = 1. - x
            y_flipped = 1. - y
            val = np.array([x, y, x_flipped, y_flipped])
            self.__vals.append(val)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2. * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big)**2 + (y - self.r_big)**2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles:
            return 2
        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big)**2 + (y - self.r_big)**2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big)**2 + (y - self.r_big)**2)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        if self.transform:
            sample = (self.transform(sample[0]), sample[1])
        return sample

    def __len__(self):
        return len(self.__cs)


def transform_single_input(x, num_steps, min_t, max_t, bias_t):
    x = x * (max_t-min_t) + min_t
    idx = x.searchsorted(bias_t)
    x = np.insert(x, idx, bias_t)
    x = torch.Tensor(x)
    x = torch.round((num_steps/max_t)*x)
    x = F.one_hot(x.to(torch.int64), num_steps + 1)
    x = x.to(torch.float32)
    return x.squeeze().T


class SpikeTimeTransform(object):
    """Transform inputs into spike times, given min, max, and bias times

    Args:
        num_steps (int): Number of time steps; i.e., (max_t - min_t)/num_steps is dt
        min_t (int): Minimum time
        max_t (int): Maximum time
        orig_max (int): The maximum value of the original data
        orig_min (int): The minimum value of the original data
    """
    def __init__(self, num_steps, min_t, max_t, bias_t):
        self.num_steps = num_steps
        self.min_t = min_t
        self.max_t = max_t
        self.bias_t = bias_t
        
    
    def __call__(self, x):
        return transform_single_input(x,
                                      self.num_steps, 
                                      self.min_t,
                                      self.max_t,
                                      self.bias_t)
    
num_steps = 1000
min_t = 0.15
max_t = 2.0
bias_t = 0.9

dataset_train = YinYangDataset(size=5000, 
                               seed=42, 
                               transform=SpikeTimeTransform(num_steps, min_t, max_t, bias_t)
                              )
dataset_test = YinYangDataset(size=1000, 
                              seed=40,
                              transform=SpikeTimeTransform(num_steps, min_t, max_t, bias_t)
                             )

if args.batch_size:
    batchsize_train = args.batch_size
else:
    batchsize_train = 150
batchsize_eval = len(dataset_test)

train_loader = DataLoader(dataset_train, batch_size=batchsize_train, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=batchsize_eval, shuffle=False)

# Hyperparams
num_inputs = 5
num_hidden = 120
num_classes = 3

# alpha, beta, theta need to rescale for loss vs. network
BETA = 1.0
ALPHA = 0.99
THETA = 1.0

dt = max_t/num_steps

# network params
ALPHA_S = np.exp(-dt * ALPHA)
BETA_S = np.exp(-dt * BETA)
THETA_S = THETA * (num_steps/max_t)

# loss hyper params
TAU_0 = 1.0
TAU_1 = 1.0
GAMMA = 0

# optimizer params
BETA_1 = 0.9
BETA_2 = 0.999
EPS = 1e-8
ETA = 0.0005

if args.spike_grad:
    if args.spike_grad == 'atan':
        spike_grad = snn.surrogate.atan(alpha=2.0)
    elif args.spike_grad == 'fast':
        spike_grad = snn.surrogate.fast_sigmoid(slope=25)
    elif args.spike_grad == 'sigmoid':
        spike_grad = snn.surrogate.sigmoid(slope=25)
    elif args.spike_grad == 'sre':
        spike_grad = snn.surrogate.spike_rate_escape(beta=1, slope=25)
    elif args.spike_grad == 'ste':
        spike_grad = snn.surrogate.straight_through_estimator()
    elif args.spike_grad == 'tri':
        spike_grad = snn.surrogate.triangular()
    else:
        spike_grad = snn.surrogate.sigmoid(slope=25)
else:
    spike_grad = snn.surrogate.sigmoid(slope=25)

net = Net(
    num_inputs,
    num_hidden,
    num_classes, 
    num_steps, 
    ALPHA_S, 
    BETA_S, 
    THETA_S,
    layer_means=[1.5, 2.0],
    layer_stds=[0.8, 0.1],
    spike_grad=spike_grad,
)

optimizer = torch.optim.Adam(
    net.parameters(), 
    lr=ETA,
    betas=(BETA_1, BETA_2),
    eps=EPS,
)

loss_fn = SF.ce_temporal_loss()

# set training parameters
if args.n_epochs:
    n_epochs = args.n_epochs
else:
    n_epochs = 1
train_accuracies = []
train_losses = []

def forward_batch(bsz, data, labels, model, loss):
    spk1, mem1, spk2, mem2 = model(torch.swapaxes(data, 0, 1))
    
    acc = SF.accuracy_temporal(spk2, labels.squeeze())
    l = loss(spk2, labels.squeeze())
    
    return acc, l

start_time = time.perf_counter()

for k in range(n_epochs):
    for i, (batch, batch_labels) in enumerate(train_loader):

        bsz = len(batch)
        
        batch_acc, batch_loss = forward_batch(bsz, batch, batch_labels, net, loss_fn)

        train_accuracies.append(batch_acc)
        train_losses.append(batch_loss)

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()  # take optimizer step


        if i % 10 == 0:
            end_time = time.perf_counter()
            print("Time elapsed (sec)=", end_time - start_time)
            print("epoch=", k)
            print("loss=", batch_loss)
            print("acc=", batch_acc)

print('evaluating...')

eval_acc = []
eval_losses = []

for i, (batch, batch_labels) in enumerate(test_loader):
    bsz = len(batch)

    batch_acc, batch_loss = forward_batch(bsz, batch, batch_labels, net, loss_fn)
    
    eval_acc.append(batch_acc)
    eval_losses.append(batch_loss)

    
train_losses = torch.stack(train_losses, dim=-1)
eval_losses = torch.stack(eval_losses, dim=-1)

print(eval_acc)

with open('train_eval_acc_' + args.spike_grad + '.npy', 'wb') as f:
    np.save(f, train_losses.detach().numpy())
    np.save(f, eval_losses.detach().numpy())
    np.save(f, np.array(eval_acc))
