import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing



def plot_learning_curves(costs_train, costs_test, m_examples):
    plt.style.use("seaborn")
    plt.plot(m_examples, costs_train, label='Training cost')
    plt.plot(m_examples, costs_test, label='Cross-validation cost')
    plt.ylabel('Cost', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()





def plot_results_bayes(bayes_trials):
    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
    result = bayes_trials_results[0]
    hyper_params = result["hyper_params"]
    cleaned_result = {"loss": result["loss"], "cv_accuracy": result["cv_accuracy"], "network": [hyper_params["hidden_layers"]["network"]], "learning_rate": hyper_params["learning_rate"], "iterations": hyper_params["iterations"], "reg_param": hyper_params["reg_param"]}

    df = pd.DataFrame.from_dict(cleaned_result)
    table = plt.table(cellText=df.values, colLabels=df.columns, colWidths = [0.15]*len(df.columns), cellLoc = 'center', rowLoc = 'center', loc='top')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.axis('off')
    plt.show()

    


def plot_results_random(result_random):
    hyper_params = result_random["hyper_params"]
    cleaned_result = {"loss": 1 - result_random["cv_accuracy"], "cv_accuracy": result_random["cv_accuracy"], "network": [hyper_params["activation_layers"]], "learning_rate": hyper_params["learning_rate"], "iterations": hyper_params["iterations"], "reg_param": hyper_params["reg_param"]}
    df = pd.DataFrame.from_dict(cleaned_result)
    table = plt.table(cellText=df.values, colLabels=df.columns, colWidths = [0.15]*len(df.columns), cellLoc = 'center', rowLoc = 'center', loc='top')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.axis('off')
    plt.show()
