# ICASSP 2025 Signal Processing Grand Challenge
This README provides a concise guide to reproducing the experiments for the ICASSP 2025 Signal Processing Grand Challenge. This guide will walk you through setting up the environment, training the model, and evaluating it on the test set.
## Track1
### Prerequisites

Ensure you have Python 3.8+ installed. You can install all required dependencies by running:

```sh
pip install -r requirements.txt
```

### Configuration

The parameters for training and evaluation are managed in `config.yaml`. Edit this file to fit your environment or requirements before running any scripts.

### Training the Model

To train the model from scratch:

1. Set up the training dataset using the `AccelDataset` class defined in `dataset.py`. Ensure your data is properly organized.
2. Run the `run.py` script:

   ```sh
   python run.py
   ```

This script will initialize the model, set up the optimizer, and train the model using the `Trainer` class. Training logs are saved using TensorBoard.

### Testing the Model

After training, use `test.py` to evaluate the model's performance:

```sh
python test.py
```

This script will load the saved model, evaluate it on the test set, and save the predictions in the `results` folder.

### Evaluating the Model

Evaluation metrics such as Balanced Accuracy (Acc_balanced) are computed using True Positive Rate (TPR) and True Negative Rate (TNR). Use the `evaluator()` function in `trainer.py` for additional evaluation and anomaly scoring.

### Logging and Checkpoints

- **Logs**: Saved in the `runs/` directory for monitoring training and validation.
- **Checkpoints**: Models are saved periodically during training based on performance improvements.

You can monitor the training process using TensorBoard:

```sh
tensorboard --logdir=runs/
```

---

## Track2
### Prerequisites

Ensure you have Python 3.8+ installed. You can install all required dependencies by running:

```sh
pip install -r requirements.txt
```

### Configuration

The parameters for training and evaluation are managed in `config.yaml`. Edit this file to fit your environment or requirements before running any scripts.

### Training the Model

To train the model from scratch:

1. Set up the training dataset using the `AccelDataset` class defined in `dataset.py`. Ensure your data is properly organized.
2. Run the `run.py` script:

   ```sh
   python run.py
   ```

This script will initialize the model, set up the optimizer, and train the model using the `Trainer` class. Training logs are saved using TensorBoard.

### Testing the Model

After training, use `test.py` to evaluate the model's performance:

```sh
python test.py
```

This script will load the saved model, evaluate it on the test set, and save the predictions in the `results` folder.

### Logging and Checkpoints

- **Logs**: Saved in the `runs/` directory for monitoring training and validation.
- **Checkpoints**: Models are saved periodically during training based on performance improvements.

You can monitor the training process using TensorBoard:

```sh
tensorboard --logdir=runs/
```
---
## License
This project is released under the CC BY-NC-ND 4.0 license.

