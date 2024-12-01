import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from loss import ASDLoss
import utils
from torch.utils.data import DataLoader
from dataset import AccelDataset

class Trainer:
    def __init__(self, *args, **kwargs):
        # Initialize training parameters
        self.args = kwargs['args']
        self.net = kwargs['net']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = self.args.writer
        self.logger = self.args.logger
        self.criterion = ASDLoss().to(self.args.device)

    def train(self, train_loader):
        # Create directory to save models
        model_dir = os.path.join(self.writer.log_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        epochs = self.args.epochs
        num_steps = len(train_loader)
        best_metric = 0
        no_better_epoch = 0

        # Training loop
        for epoch in range(0, epochs + 1):
            sum_loss = 0
            self.net.train()
            train_bar = tqdm(train_loader, total=num_steps, desc=f'Epoch-{epoch}')
            
            for (x_xwavs, x_ywavs, x_zwavs, x_xmels, x_ymels, x_zmels, labels) in train_bar:
                # Data preparation
                x_xwavs, x_ywavs, x_zwavs = x_xwavs.float().to(self.args.device), x_ywavs.float().to(self.args.device), x_zwavs.float().to(self.args.device)
                x_xmels, x_ymels, x_zmels = x_xmels.float().to(self.args.device), x_ymels.float().to(self.args.device), x_zmels.float().to(self.args.device)
                labels = labels.reshape(-1).long().to(self.args.device)

                # Forward pass and loss calculation
                logits, _ = self.net(x_xwavs, x_ywavs, x_zwavs, x_xmels, x_ymels, x_zmels, labels)
                loss = self.criterion(logits, labels)
                train_bar.set_postfix(loss=f'{loss.item():.5f}')

                # Backward pass and parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log loss value for visualization
                self.writer.add_scalar(f'train_loss', loss.item(), self.sum_train_steps)
                sum_loss += loss.item()
                self.sum_train_steps += 1

            # Average loss calculation
            avg_loss = sum_loss / num_steps
            if self.scheduler is not None and epoch >= self.args.start_scheduler_epoch:
                self.scheduler.step()

            self.logger.info(f'Epoch-{epoch}\tloss:{avg_loss:.5f}')
            
            # Validation and early stopping
            if (epoch - self.args.start_valid_epoch) % self.args.valid_every_epochs == 0 and epoch >= self.args.start_valid_epoch:
                acc_balanced = self.test(save=False)
                self.writer.add_scalar(f'auc', acc_balanced, epoch)
                
                if acc_balanced >= best_metric:
                    no_better_epoch = 0
                    best_metric = acc_balanced
                    best_model_path = os.path.join(model_dir, 'best_checkpoint.pth.tar')
                    utils.save_model_state_dict(best_model_path, epoch=epoch, net=self.net.module if self.args.dp else self.net, optimizer=None)
                    self.logger.info(f'Best epoch now is: {epoch:4d}')
                else:
                    no_better_epoch += 1
                    if no_better_epoch > self.args.early_stop_epochs:
                        break
            
            # Save model at intervals
            if epoch >= self.args.start_save_model_epochs and (epoch - self.args.start_save_model_epochs) % self.args.save_model_interval_epochs == 0:
                model_path = os.path.join(model_dir, f'{epoch}_checkpoint.pth.tar')
                utils.save_model_state_dict(model_path, epoch=epoch, net=self.net.module if self.args.dp else self.net, optimizer=None)

    def test(self, save=False):
        """
        Test the model on the test dataset and output predictions to a CSV file.
        """
        csv_output = []
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        
        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)

        test_dataset = AccelDataset(self.args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        y_true, y_pred = [], []

        for idx, (x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, label) in enumerate(test_loader):
            # Data transfer to the device
            x_xwav, x_ywav, x_zwav = x_xwav.to(self.args.device), x_ywav.to(self.args.device), x_zwav.to(self.args.device)
            x_xmel, x_ymel, x_zmel = x_xmel.to(self.args.device), x_ymel.to(self.args.device), x_zmel.to(self.args.device)

            # Forward pass for predictions
            with torch.no_grad():
                predict_ids, _ = net(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                predicted_label = torch.argmax(predict_ids, dim=1).cpu().item()
                y_pred.append(predicted_label)
                y_true.append(label.cpu().item())

            # Save chunk_id and label to CSV output
            chunk_id = idx
            csv_output.append([chunk_id, predicted_label])

        # Calculate balanced accuracy (TPR + TNR)
        TP = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
        TN = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0)
        FP = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
        FN = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

        acc_balanced = 0.5 * (TPR + TNR)
        self.logger.info(f'Acc_balanced: {acc_balanced:.3f}')

        # Save predictions to CSV if needed
        if save:
            csv_path = os.path.join(result_dir, 'test_predictions.csv')
            with open(csv_path, 'w') as f:
                f.write("chunk_id,label\n")
                for chunk_id, label in csv_output:
                    f.write(f"{chunk_id},{label}\n")
            self.logger.info(f"Test results saved to {csv_path}")
        
        return acc_balanced


    # Evaluate the model and compute anomaly scores for all test files
    def evaluator(self, save=True, gmm_n=False):
        result_dir = os.path.join('./evaluator/teams', self.args.version)
        if gmm_n:
            result_dir = os.path.join('./evaluator/teams', self.args.version + f'-gmm-{gmm_n}')
        os.makedirs(result_dir, exist_ok=True)

        self.net.eval()
        net = self.net.module if self.args.dp else self.net
        print('\n' + '=' * 20)
        
        # Iterate over test and train directories
        for index, (target_dir, train_dir) in enumerate(zip(sorted(self.args.test_dirs), sorted(self.args.add_dirs))):
            machine_type = target_dir.split('/')[-2]
            # Get list of machines
            machine_id_list = utils.get_machine_id_list(target_dir)
            
            for id_str in machine_id_list:
                meta = machine_type + '-' + id_str
                label = self.args.meta2label[meta]
                test_files = utils.get_filename_list(target_dir, pattern=f'{id_str}*')
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{id_str}.csv')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                
                if gmm_n:
                    # Extract features for GMM fitting
                    train_files = utils.get_filename_list(train_dir, pattern=f'normal_{id_str}*')
                    features = self.get_latent_features(train_files)
                    means_init = net.arcface.weight[label * gmm_n: (label + 1) * gmm_n, :].detach().cpu().numpy() \
                        if self.args.use_arcface and (gmm_n == self.args.sub_center) else None
                    gmm = self.fit_GMM(features, n_components=gmm_n, means_init=means_init)
                
                # Process test files to compute anomaly scores
                for file_idx, file_path in enumerate(test_files):
                    x_wav, x_mel, label = self.transform(file_path)
                    x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(self.args.device)
                    label = torch.tensor([label]).long().to(self.args.device)
                    with torch.no_grad():
                        predict_ids, feature = net(x_wav, x_mel, label)
                    
                    # Compute anomaly scores using GMM or softmax probabilities
                    if gmm_n:
                        if self.args.use_arcface: feature = F.normalize(feature).cpu().numpy()
                        y_pred[file_idx] = - np.max(gmm._estimate_log_prob(feature))
                    else:
                        probs = - torch.log_softmax(predict_ids, dim=1).mean(dim=0).squeeze().cpu().numpy()
                        y_pred[file_idx] = probs[label]
                    
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                
                # Save the results to a CSV file
                if save:
                    utils.save_csv(csv_path, anomaly_score_list)

    # Extract and normalize latent features from training files
    def get_latent_features(self, train_files):
        pbar = tqdm(enumerate(train_files), total=len(train_files))
        self.net.eval()
        classifier = self.net.module if self.args.dp else self.net
        features = []
        
        # Process each training file to extract features
        for file_idx, file_path in pbar:
            x_wav, x_mel, label = self.transform(file_path)
            x_wav, x_mel = x_wav.unsqueeze(0).float().to(self.args.device), x_mel.unsqueeze(0).float().to(self.args.device)
            label = torch.tensor([label]).long().to(self.args.device)
            with torch.no_grad():
                _, feature, _ = classifier(x_wav, x_mel, label)
            if file_idx == 0:
                features = feature.cpu()
            else:
                features = torch.cat((features.cpu(), feature.cpu()), dim=0)
        
        # Normalize features if using arcface
        if self.args.use_arcface: features = F.normalize(features)
        return features.numpy()

    # Fit a GMM model using training data
    def fit_GMM(self, data, n_components, means_init=None):
        print('=' * 40)
        print('Fitting GMM for test data...')
        np.random.seed(self.args.seed)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              means_init=means_init, reg_covar=1e-3, verbose=2)
        gmm.fit(data)
        print('GMM fitting completed.')
        print('=' * 40)
        return gmm
