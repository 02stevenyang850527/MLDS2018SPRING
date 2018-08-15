# Codes usage:

## Deep vs Shallow

### HW1-1.1: Simulate a function.
Follow the command below to finish training and plotting prediction and loss.
1. Simulate damping function:
```
python3 hw1_1_1.py --func damping --model shallow
python3 hw1_1_1.py --func damping --model medium
python3 hw1_1_1.py --func damping --model deep
python3 plot.py --mode hw1-1-1 --func damping
```
2. Simulate triangle wave function:
```
python3 hw1_1_1.py --func triangle --model shallow
python3 hw1_1_1.py --func triangle --model medium
python3 hw1_1_1.py --func triangle --model deep
python3 plot.py --mode hw1-1-1 --func triangle
```

### HW1-1.2: Train on actual task using deep and shallow models
```
python3 hw1_1_2.py
```
Bonus: Train on different task (CIFAR-10)
```
python3 hw1_1_2_cifar.py
```
Note: Variable `isCNN = True` in `hw1_1_2.py` and `hw1_1_2_cifar.py` is for experiment about CNN; otherwise for experiment about DNN. Both experiments are based on MNIST dataset.  

## Optimization

### HW1-2.1: Visualize the optimization process
First, run following command and assign the no. of training events to have multiple models and the related training process.
```
python3 hw1_2_1_model.py [no. of events]
```
Second, plot it! (Change variable `num_event` in `hw1_2_1_parse.py` if necessary.)
```
python3 hw1_2_1_parse.py
```
Note: Variable `isFull = True` in `hw1_1_2_1_parse.py` perform dimension reduction on all parameters in the model; otherwise, the parameters in the first hidden layer.

### HW1-2.2: Observe gradient norm
1.Applied sinc function:
Follow the command below to finish training and plotting its loss and gradient norm.
```
python3 hw1_2_2_func.py
```
2.Applied MNIST data
Follow the command below to finish training and plotting its loss and gradient norm.
```
python3 hw1_2_2_mnist.py
```

### HW1-2.3: Calculate minimal ratio at gradient equal to 0
```
python3 hw1_2_3.py
python3 hw1_2_3.py
... 
python3 hw1_2_3.py
python3 plot.py --mode hw1-2-3
```
Note: You can execute `hw1_2_3.py` many times as you wish and plot all minimal ratio together. When epoch > 50,000, the program will shut down automatically.

### HW1-2 bonus: Visualize the error surface
```
python3 hw1_2_bonus.py
```

## Generalization

### HW1-3.1: Can network fit random labels?
```
python3 hw1_3_1.py
```

### HW1-3.2: Number of parameters vs Generalization
Follow the command below to finish training 11 CNN models with differnent parameter numbers. (cifar10 is applied as the dataset.)
```
python3 hw1_3_2_cifar10.py
```

### HW1-3.3: Flateness vs Generalization
Part1: Run following command.
```
python3 hw1_3_3_part1.py
```
Part2:  
```
python3 hw1_3_3_part2.py --batch 128
python3 hw1_3_3_part2.py --batch 256
...
python3 hw1_3_3_part2.py --batch 4096
python3 plot.py hw1-3-3-part2
```
Note: You can execute `hw1_3_3_part2.py` in different batch size and plot all sensitivity together.
