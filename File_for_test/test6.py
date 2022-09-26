import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as atlas
import torch
import numpy as np


y_true = ["bat", "ball", "ball", "bat", "bat", "bat"]
y_pred = ["bat", "bat", "ball", "ball", "bat", "bat"]
y_true = torch.tensor([0, 0, 1, 1, 0, 0])
y_pred = torch.tensor([0, 1, 1, 1, 0, 0])
conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='turbo')

# labels the title and x, y axis of plot
fx.set_title('Confusion Matrix');
fx.set_xlabel('Predicted')
fx.set_ylabel('Actual');

# labels the boxes
fx.xaxis.set_ticklabels(['bat','ball'])
fx.yaxis.set_ticklabels(['bat','ball'])

atlas.show()