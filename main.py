#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import GCN
from graph_builder import build_karate_club_graph, draw

import warnings
warnings.filterwarnings("ignore")


# Model
net = GCN(34, 5, 2)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
print("Model structure:", net)

# Data
G = build_karate_club_graph()
inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])
labels = torch.tensor([0, 1])


# Train
net.train()
for epoch in range(30):
    optimizer.zero_grad()

    # 前向传播
    logits = net(G, inputs)

    all_logits.append(logits.detach())

    # Softmax
    logp = F.log_softmax(logits, 1)

    # 只计算了0 33两个有label的node的损失
    loss = F.nll_loss(logp[labeled_nodes], labels)

    # 反向传播
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch + 1, loss.item()))


# Generate figure
for i in range(0, 30):
    fig = plt.figure(dpi=500)
    fig.clf()
    ax = fig.subplots()
    draw(i, all_logits, G, ax)
    plt.savefig("./Epoch/Epoch"+str(i))