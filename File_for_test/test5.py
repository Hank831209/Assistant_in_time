# imports
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch

# creates confusion matrix
# y_true = ["bat", "ball", "ball", "bat", "bat", "bat"]
# y_pred = ["bat", "bat", "ball", "ball", "bat", "bat"]
y_true = torch.tensor([1, 2, 3, 0])
y_pred = torch.tensor([1, 2, 3, 1])
# mat_con = (confusion_matrix(y_true, y_pred, labels=["bat", "ball"]))
mat_con = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
print(mat_con)
# Setting the attributes
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(mat_con.shape[0]):
    for n in range(mat_con.shape[1]):
        px.text(x=m, y=n, s=mat_con[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

