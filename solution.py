import numpy as np
import matplotlib.pyplot as plt
import time

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################

######## HELPER FUNCTIONS #################
def minkowski_mat(x, Y):
    return (np.sum((np.abs(x - Y)) ** 2, axis=1)) ** (1.0 / 2)

def minkowski_mat_2(x, Y, p=2):
    return np.sum((np.abs(x - Y)) ** 2, axis=1)

def gaussian_kernel(x, sigma, d):
    second_part = -0.5 * ((x**2)/(sigma**2))
    return np.exp(second_part)

###########################################

class Q1:

    def feature_means(self, banknote):
        return np.mean(banknote[:,:-1],axis=0)

    def covariance_matrix(self, banknote):
        return np.cov(np.transpose(banknote[:,:-1]))

    def feature_means_class_1(self, banknote):
        return np.mean(banknote[banknote[:,4]==1,:-1],axis=0)

    def covariance_matrix_class_1(self, banknote):
        return np.cov(np.transpose(banknote[banknote[:,4]==1,:-1]))


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)

        # For each test datapoint
        for (i, ex) in enumerate(test_data):
            # i is the row index
            # ex is the i'th row

            distances = minkowski_mat(ex, self.train_inputs) 
            
            neighbour_idx = []
            neighbour_idx = np.array([j for j in range(len(distances)) if distances[j] < self.h])
            if len(neighbour_idx) == 0:
                classes_pred[i] = draw_rand_label(ex, np.unique(self.train_labels))
            else:
                for k in neighbour_idx:
                    counts[i, (self.train_labels[k]).astype(int)] += 1
                classes_pred[i] = np.argmax(counts[i, :]) 
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.n_classes = len(np.unique(train_labels))

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.n_classes))
        classes_pred = np.zeros(num_test)
        num_features = test_data.shape[1]
        
        # For each test datapoint
        for (i, ex) in enumerate(test_data):
            # i is the row index
            # ex is the i'th row

            distances = minkowski_mat(ex, self.train_inputs) 
            
            for k in range(len(distances)):
                counts[i, (self.train_labels[k]).astype(int)] += gaussian_kernel(distances[k], self.sigma, num_features)
            classes_pred[i] = np.argmax(counts[i, :]) 
        return classes_pred


def split_dataset(banknote):
    indexes = np.arange(banknote.shape[0])
    train_idx = np.concatenate((indexes[indexes%5==0],indexes[indexes%5==1],indexes[indexes%5==2]),axis=0)
    valid_idx = indexes[indexes%5==3]
    test_idx = indexes[indexes%5==4]
    
    return banknote[train_idx,:],banknote[valid_idx,:],banknote[test_idx,:]


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        model_hp = HardParzen(h)
        model_hp.train(self.x_train, self.y_train)
        predictions = model_hp.compute_predictions(self.x_val)
        return ((self.y_val != predictions).mean())

    def soft_parzen(self, sigma):
        model_sp = SoftRBFParzen(sigma)
        model_sp.train(self.x_train, self.y_train)
        predictions = model_sp.compute_predictions(self.x_val)
        return ((self.y_val != predictions).mean())


def get_test_errors(banknote, with_plot=False):    
    h_values = [0.01,0.1,0.2,0.3,0.4,0.5,1.,3.,10.,20.]
    sigma_values = [0.01,0.1,0.2,0.3,0.4,0.5,1.,3.,10.,20.]
    train_data, valid_data, test_data = split_dataset(banknote)
    train_inputs = train_data[:,:-1]
    train_labels = train_data[:,-1]
    valid_inputs = valid_data[:,:-1]
    valid_labels = valid_data[:,-1]
    test_inputs = test_data[:,:-1]
    test_labels = test_data[:,-1]
    error_hp = np.zeros(len(h_values))
    error_sp = np.zeros(len(sigma_values))
    
    for (i, h) in enumerate(h_values):
        error_hp_model = ErrorRate(train_inputs, train_labels, valid_inputs, valid_labels)
        error_hp[i] = error_hp_model.hard_parzen(h)
        
    for (i, sigma) in enumerate(sigma_values):
        error_sp_model = ErrorRate(train_inputs, train_labels, valid_inputs, valid_labels)
        error_sp[i] = error_sp_model.soft_parzen(sigma)
         
    h_star_idxs = np.argmin(error_hp)
    sigma_star_idxs = np.argmin(error_sp)

    h_star = h_values[h_star_idxs]
    sigma_star = sigma_values[sigma_star_idxs]
    
    error_hp_model = ErrorRate(train_inputs, train_labels, test_inputs, test_labels)
    error_h_star = error_hp_model.hard_parzen(h_star)
        
    error_sp_model = ErrorRate(train_inputs, train_labels, test_inputs, test_labels)
    error_sigma_star = error_sp_model.soft_parzen(sigma_star)
    
    if with_plot:
        plt.figure()
        plt.plot(h_values, error_hp)
        plt.plot(sigma_values, error_sp)
        plt.xlabel('h and h values')
        plt.ylabel('error rate')
        plt.legend(['Hard Parzen','Soft Parzen'])
    
    return error_h_star, error_sigma_star

def get_test_errors_projections(banknote, with_plot=False, number_of_projections=500, with_e_bars=False):  
    start = time.time()
    h_sigma_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1., 3., 10., 20.]
    train_data, valid_data, test_data = split_dataset(banknote)
    train_inputs = train_data[:,:-1]
    train_labels = train_data[:,-1]
    valid_inputs = valid_data[:,:-1]
    valid_labels = valid_data[:,-1]
    all_error_hp_projections = []
    all_error_sp_projections = []
    
    for _ in range(number_of_projections):
        error_hp = np.zeros(len(h_sigma_values))
        error_sp = np.zeros(len(h_sigma_values))
        A = np.random.normal(size=(4,2))
        
        train_projected = random_projections(train_inputs, A)
        valid_projected = random_projections(valid_inputs, A)
        
        error_model = ErrorRate(train_projected, train_labels, valid_projected, valid_labels)
        for (i, val) in enumerate(h_sigma_values):
            error_hp[i] = error_model.hard_parzen(val)
            error_sp[i] = error_model.soft_parzen(val)
                        
        all_error_hp_projections.append(error_hp)
        all_error_sp_projections.append(error_sp)
                        
    average_error_hp_projections = np.vstack(all_error_hp_projections).mean(axis=0)
    average_error_sp_projections = np.vstack(all_error_sp_projections).mean(axis=0)
            
    if with_plot:
        plt.figure()
        #plt.plot(h_sigma_values, average_error_hp_projections)
        print(np.vstack(all_error_hp_projections))
        if with_e_bars:
            plt.errorbar(h_sigma_values,
                         y=average_error_hp_projections,
                         yerr= 0.2*np.vstack(all_error_hp_projections).std(axis=0))
        
        #plt.plot(h_sigma_values, average_error_sp_projections)
        if with_e_bars:
            plt.errorbar(h_sigma_values,
                         y=average_error_sp_projections,
                         yerr= 0.2*np.vstack(all_error_sp_projections).std(axis=0))
        
        plt.xlabel('h and sigma values')
        plt.ylabel('error rate')
        plt.legend(['Hard Parzen','Soft Parzen'])
        
    print(time.time() - start)
        
def random_projections(X, A):
    return (1./np.sqrt(2))*np.dot(X,A)