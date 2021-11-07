# Bayesian_NN_mnist
``` python
python classify_mnist.py --n_train 100
```

# Even on few training samples, BNN (bayesian neural network) did not show the over fitting.
![Learning curve](classify_mnist_learning_curve_n_train_100.png?raw=true "Learning curve")
![Learning curve](classify_mnist_learning_curve_n_train_1000.png?raw=true "Learning curve")
![Learning curve](classify_mnist_learning_curve_n_train_10000.png?raw=true "Learning curve")
![Learning curve](classify_mnist_learning_curve_n_train_20000.png?raw=true "Learning curve")
![Learning curve](classify_mnist_learning_curve_n_train_40000.png?raw=true "Learning curve")

# Dependencies

```
argparse
matplotlib
numpy
pyro
torch
torchvision
tqdm
mngs
```




# Reference
https://pyro.ai/examples/modules.html
