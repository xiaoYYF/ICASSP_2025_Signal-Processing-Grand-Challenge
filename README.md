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


---

## Track 2

### Prerequisites
Make sure you have Python 3.8+ installed. You can install all required dependencies by running the following command:

```sh
pip install -r requirements.txt
```

### Configuration
The parameters for training and evaluation are managed in the `config.yaml` file. Please edit this file according to your environment or specific requirements before running any scripts.

### Training and Testing the Model

#### Training and Testing the Model
Both training and testing are handled within the `run.py` script. Follow these steps:

1. First, set up the training dataset using the `AccelDataset` class defined in `dataset.py`. Ensure your data is properly organized.
2. Run the `run.py` script:

   ```sh
   python run.py
   ```

This script will initialize the model, set up the optimizer, and start the training process. The training logs will be saved and can be monitored using TensorBoard.

The testing process is also included in `run.py`. After training, the model will be evaluated on the test set, and the predictions will be saved.

### Logging and Checkpoints
Checkpoints: We have uploaded the checkpoints used during the submission of our model. 
These checkpoints represent the model states at the time of submission and can be used for further evaluation or fine-tuning.




---
## License
This project is released under the CC BY-NC-ND 4.0 license.

