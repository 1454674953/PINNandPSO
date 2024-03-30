## Development Log

`2023/11/04`

* Add Dynamic learning rate.
* Add `train.py`and`pred.py` in v1.

## Description of the document

> `test_pred_GPU` 
>
>The python file only has a prediction function, which uses a neural network for prediction.
>This python file is a basic reference for other files that use neural networks.
>And this code adds the use of GPU training function, but the result is not good, 
>the use of CPU training code is used in the front of the other code.

> `test_no_features` 
> 
> This file integrates neural networks with the problem of predicting optimal solutions, 
> first using the training code in `test_pred_GPU.py`, 
> and then training a model on top of that, 
> using a gradient descent algorithm to optimize the feature values through backpropagation so that they can reach or approach the target output value. 
> The three feature values at the beginning of this code are randomized.

> `test_give_features`
> 
> This file is based on `test_no_features.py`, given two of the eigenvalues, 
> randomize the remaining eigenvalues as initial inputs, and then train them,
> this code is problematic, v2 below is fine.

> `test_give_features_2`
> 
> `test_give_features.py` **version_2**, this code is effective in predicting eigencalues.

### v1 document

> `train` train the model using gradient descent.

> `train_v2` train the model using PSO, **_but it's not working well_**.

> `pred` prediction the model using gradient descent.

> `pred_v2` prediction the model using PSO.

### Dataest
`Advertising.csv`

### Aims
* NN + backward propagation (initial completion)
* PINN/DeepXDE + backward propagation (initial completion)
* NN/PINN/DeepXDE + PSO and so on... (initial completion)

### In the future, we develop 
* The predicted eigenvalues should have a function to limit the range 
* Currently an output value is given, later optimization our goal is to let it find the optimal value by itself.