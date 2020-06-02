import data_processing as dproc
from sklearn.model_selection import train_test_split
import model
import dataplot
import numpy as np
from hyperopt import hp
from hyperopt import tpe
from hyperopt import fmin
from hyperopt import STATUS_OK
from hyperopt.pyll.base import scope
from hyperopt import Trials



X_train_std = None
X_cv_std = None
y_train = None
y_cv = None



def heart_disease(data_file, iterations=3000, learning_rate=0.1, reg_param=0.1, tuning=False, plot_learning_curves=False, plot_results=False):
    global X_train_std, X_cv_std, y_train, y_cv
    
    dataset = dproc.read_dataset(data_file)
    X, y = dproc.preprocess(dataset)
    X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3)
    
    # Standardize data
    X_train_std, X_cv_std = dproc.standardize(X_train, X_cv)

   
    if tuning == "random":
        random_tune(plot_results)
    elif tuning == "bayes":
        bayes_tune(plot_results)
    else:
        train_manually(iterations, learning_rate, reg_param, plot_learning_curves)
    


    
        








def train_manually(iterations, learning_rate, reg_param, plot_learning_curves):
    activation_layers = (X_train_std.shape[1], 25, 1)
    parameters, V = model.init_params_and_V(activation_layers)

    model.train_model(X_train_std.T, y_train.T, parameters, V, iterations, learning_rate, reg_param)

    train_accuracy = model.compute_accuracy(X_train_std.T, y_train.T, parameters)
    cv_accuracy = model.compute_accuracy(X_cv_std.T, y_cv.T, parameters)
    
    print(f"Train accuracy: {train_accuracy}")
    print(f"CV accuracy: {cv_accuracy}")


    if plot_learning_curves:
        costs_train, costs_cv, m_examples = model.train_various_sizes(X_train_std.T, X_cv_std.T, y_train.T, y_cv.T, parameters, V, activation_layers, 3000, 0.01, reg_param)
        dataplot.plot_learning_curves(costs_train, costs_cv, m_examples)




        
        
def random_tune(plot_results):
    accuracies_parameters_hyperparameters = {"cv_accuracy": 0}

    # "Random" hyperparameter tuning
    for i in range(50):
        activation_layers = (X_train_std.shape[1], np.random.randint(13, 31), 1)
        parameters, V = model.init_params_and_V(activation_layers)
        iterations = np.random.randint(1000, 5001)
        learning_rate = 10**(-1 - 2 * np.random.rand())
        reg_param = 10**(-2 * np.random.rand())

        model.train_model(X_train_std.T, y_train.T, parameters, V, iterations, learning_rate, reg_param)

        train_accuracy = model.compute_accuracy(X_train_std.T, y_train.T, parameters)
        cv_accuracy = model.compute_accuracy(X_cv_std.T, y_cv.T, parameters)

        if cv_accuracy > accuracies_parameters_hyperparameters["cv_accuracy"]:
            accuracies_parameters_hyperparameters = {"train_accuracy": train_accuracy, "cv_accuracy": cv_accuracy, "params": parameters, "hyper_params": {"activation_layers": activation_layers, "iterations": iterations, "learning_rate": learning_rate, "reg_param": reg_param}}


        print(f"Train accuracy: {train_accuracy}")
        print(f"CV accuracy: {cv_accuracy}\n")

    if plot_results:
        dataplot.plot_results_random(accuracies_parameters_hyperparameters)
        
    print(f"Train accuracy: {accuracies_parameters_hyperparameters['train_accuracy']}")
    print(f"Best cv accuracy: {accuracies_parameters_hyperparameters['cv_accuracy']}")
    print(accuracies_parameters_hyperparameters["hyper_params"])





    

def bayes_tune(plot_results):
    space = {
        "hidden_layers": hp.choice("options", [{"hidden_layers": 1, "network": (X_train_std.shape[1], scope.int(hp.quniform("1_hidden_1", 5, 32, 1)), 1)},
                                                     {"hidden_layers": 2, "network": (X_train_std.shape[1], scope.int(hp.quniform("2_hidden_1", 5, 32, 1)), scope.int(hp.quniform("2_hidden_2", 5, 32, 1)), 1)}]),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.2)),
        "iterations": scope.int(hp.quniform("iterations", 500, 5000, 1)),
        "reg_param": hp.loguniform("reg_param", np.log(0.001), np.log(1))
    }

    bayes_trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=250, trials=bayes_trials)

    if plot_results:
        dataplot.plot_results_bayes(bayes_trials)


    

def objective(hyper_params):
    activation_layers = tuple(int(a) for a in hyper_params["hidden_layers"]["network"])
    
    learning_rate = hyper_params["learning_rate"]
    iterations = int(hyper_params["iterations"])
    reg_param = hyper_params["reg_param"]
    parameters, V = model.init_params_and_V(activation_layers)

    model.train_model(X_train_std.T, y_train.T, parameters, V, iterations, learning_rate, reg_param)
    
    train_accuracy = model.compute_accuracy(X_train_std.T, y_train.T, parameters)
    cv_accuracy = model.compute_accuracy(X_cv_std.T, y_cv.T, parameters)

    loss = 1 - cv_accuracy

    print(hyper_params)

    return {"loss": loss, "train_accuracy": train_accuracy, "cv_accuracy": cv_accuracy, "hyper_params": hyper_params, "status": STATUS_OK}


    
