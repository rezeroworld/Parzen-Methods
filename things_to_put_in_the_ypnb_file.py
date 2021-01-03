import numpy as np
import solution
import matplotlib.pyplot as plt

data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
train_inputs = data[:,:-1]
train_labels = data[:,-1]

solution.get_test_errors_projections(data, with_plot=True, number_of_projections=2, with_e_bars=True)