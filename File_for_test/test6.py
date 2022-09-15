from cProfile import label
from pretty_confusion_matrix import pp_matrix_from_data
import numpy as np

y_true = np.array([0, 1, 2, 3])
y_predict = np.array([0, 2, 2, 3])
pp_matrix_from_data(y_true, y_predict, columns=['Baby', 'Princess', 'Casual Wear', 'Gentleman'], 
                    cmap='rainbow', pred_val_axis='x')



