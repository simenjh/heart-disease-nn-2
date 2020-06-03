# Heart disease detection through neural network 2

This project is a continuation of [Heart disease detection through logistic regression 1](https://github.com/simenjh/heart-disease-regression-1), [Heart disease detection through logistic regression 2](https://github.com/simenjh/heart-disease-regression-2) and [Heart disease detection through neural network 1](https://github.com/simenjh/heart-disease-nn-1)

This neural network implementation from scratch is an attempt to improve model accuracy through different hyperparameter optimization techniques. 


## The purposes of this project:
* Compare the performance (accuracy) of manual, random and informed hyperparameter tuning methods.
* Analyze results and suggest possible improvements. 


Dataset: [Heart disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

<br />

## Run program
* Call heart_disease(data_file) in heart_disease.py.
* Pass tuning="random" or tuning="bayes" to use random search or bayes optimization.
* Drop passing the tuning argument to tune manually. 

<br /> <br />



## Manual search
As reported in [Heart disease detection through neural network 1](), tuning manually, 
I achieved 80% - 87% accuracy on the cross-validation set. 

<br />	   

## Random search
Random search works by selecting random values for the hyperparameters. This is a totally uninformed approach, learning nothing from past results. 

### Results
![](images/random_results.png?raw=true)

Above, you can see the results from a single test run of 250 random selections of hyperparameters. 
Running the random search algorithm about 50 times has resulted in cross-validation accuracies of 84% - 90%. 

 <br />

## Bayesian hyperparameter optimization
Contrary to random search, this approach considers the performance of previously selected hyperparameters when selecting which hyperparameters to try next.

Bayesian optimization finds the loss that minimizes an objective function. It does this by building a surrogate function (probability model) that is built from past evaluation results of the objective function. The surrogate function is much cheaper to evaluate than the objective function. Values returned from the surrogate function are selected using an expected improvement criterion.

The process can be described like this:
1. Build a surrogate probability model of the objective function.
2. Find the hyperparameters that perform best on the surrogate.
3. Apply these hyperparameters to the true objective function.
4. Update the surrogate model incorporating the new results.
5. Repeat steps 2–4 until max iterations or time is reached.

I have used 1 - cross validation accuracy as the the value returned by the objective function, and the loss to be minimized. 

For this project, Bayesian optimization is done with the Hyperopt library, and the surrogate used is Tree Parzen Estimator (TPE). 

### Results
![](images/bayes_results.png?raw=true)

In the table above, you can see results from running the bayesian optimizer with random initialization of data to training and cross validation sets. The accuracy on the cross-validation set is about 90%. In general it ranges from 84% - 94% depending on random initialization of data. 

<br />

## Conclusions
Both random search and bayesian optimization result in much better accuracies on the cross-validation set compared with manual tuning.

Bayesian optimization performed only a little bit better than random search. This is likely due to the small dataset, as it easier to get "lucky" with less data. 

<br />

## Possible improvements
* More data.

<br />


The dataset has 303 examples, which likely isn't enough for this problem. 
