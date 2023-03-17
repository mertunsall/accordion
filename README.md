# Accordion

To run experiments, use

```
python run_experiment config_file.json # runs experiment with ResNet architecture
python run_transformer_experiments config_file.json # runs experiment with ViT architecture
```

NOTE: There is an unresolved bug with federated transformer experiments.

The list of experiments include

1. Centralized
2. CentralizedAccordion
3. CentralizedDistillAccordion
4. CentralizedSmall 
5. Fed
6. FedAccordion
7. FedDistillAccordion
8. FedClass
9. FedSmall

which includes all the methods described in the report. All the config files are generated from files `0_exp_configs.ipynb` and `transformer_exp_configs.ipynb` files and saved into the folder `exp_configs`. Runnning the experiments save the results into files `saved_models/CIFAR10` or `saved_models/CIFAR100` depending on the dataset specified in the config file. The saved information includes:

1. `learning_rates.pkl`: an array saving the learning rates for every epoch/global communication round
2. `model` or `model_i` (for FedClass): final global model or models
3. `val_acc.pkl`: validation accuracy throughout training
4. `val_loss.pkl`: validation loss throughout training
5. `training_loss.pkl`: training loss throughout training
6. `results.json`: test accuracies for different model fractions used in the training

For centralized experiments, the last file `results.json` is created after running `get_centralized_results.py {dataset}` where the `dataset` variable is either CIFAR10 or CIFAR100. We suggest commenting the lines for the types of experiments you haven't already ran so that the script calculates the results successfully. You can change the variable `model_name` in the script to get the results for ResNet or ViT architectures.

The important Jupyter notebooks are for the following purposes:

1. `0_centralized_cifar10_results.ipynb`: Analyzes the results of centralized CIFAR10 experiments for ResNets.
2. `0_centralized_cifar100_results.ipynb`: Analyzes the results of centralized CIFAR100 experiments for ResNets.
3. `0_energy_accuracy.ipynb`: Analyzes the energy accuracy tradeoffs and some other relevant figures.
4. `0_flops_speed_energy.ipynb`: Generates relevant figures related to flops, speed (latency), and energy.
5. `0_federated_cifar10_results.ipynb`: Analyzes the results of federated CIFAR10 experiments for ResNets.
6. `0_federated_cifar100_results.ipynb`: Analyzes the results of federated CIFAR100 experiments for ResNets.
7. `0_centralized_cifar100_results.ipynb`: Tests the functionality of Accordion features applied in models coded in `models` folder.
8. `transformer_centralized_cifar10_results.ipynb`: Analyzes the results of centralized CIFAR10 experiments for ViTs.
9. `transformer_centralized_cifar100_results.ipynb`: Analyzes the results of centralized CIFAR100 experiments for ViTs.

The experiment configs can be understood through looking at examples in `exp_configs`.

