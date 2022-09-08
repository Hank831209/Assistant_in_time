# imports
import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as atlas

# y_true = ["bat", "ball", "ball", "bat", "bat", "bat"]
# y_pred = ["bat", "bat", "ball", "ball", "bat", "bat"]
labels = [0, 1, 2]
y_true = [0, 1, 1]
y_pred = [0, 1, 3]
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)
# Using Seaborn heatmap to create the plot
fx = sebrn.heatmap(conf_matrix, annot=True, cmap='turbo')

# labels the title and x, y axis of plot
fx.set_title('Plotting Confusion Matrix using Seaborn\n\n')
fx.set_xlabel('Predicted Values')
fx.set_ylabel('Actual Values ')

# labels the boxes
# fx.xaxis.set_ticklabels(labels)
# fx.yaxis.set_ticklabels(labels)

atlas.show()

