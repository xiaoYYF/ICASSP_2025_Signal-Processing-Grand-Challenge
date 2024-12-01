import os
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import AccelDataset  
from net import STgramMFN  
import utils  

class Tester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu')  # Set device (GPU or CPU)
        self._setup_model()
        self.logger = utils.get_logger(filename=os.path.join(self.args.result_dir, 'test.log'))  # Logger setup

    def _setup_model(self):
        # Initialize model
        self.net = STgramMFN(num_classes=len(self.args.meta2label), use_arcface=self.args.use_arcface,
                             m=self.args.m, s=self.args.s, sub=self.args.sub_center)
        # Handle multi-GPU setup (DataParallel)
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.args.device_ids)
        self.net = self.net.to(self.device)

        # Load pre-trained model weights
        model_path = os.path.join("/home/competition/2024ICASSPGC-8/ICASSP_GC-8_3090/runs/STgram-MFN(m=0.7,s=30) copy/model/best_checkpoint.pth.tar")
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.net.module.load_state_dict(torch.load(model_path)['model'])
        else:
            self.net.load_state_dict(torch.load(model_path)['model'])

    def test(self, save=False):
        """
        Evaluate the model on the test dataset and save predictions to a CSV file.
        """
        csv_output = []  # List to store predictions
        result_dir = os.path.join(self.args.result_dir, 'test_results')
        os.makedirs(result_dir, exist_ok=True)  # Create result directory if not exists
        
        # Set model to evaluation mode
        self.net.eval()
        net = self.net.module if self.args.device_ids and len(self.args.device_ids) > 1 else self.net
        print('\n' + '=' * 20)

        # Load test dataset
        test_dataset = AccelDataset(args)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Iterate through the test data
        for idx, (x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel) in enumerate(test_loader):
            # Move data to the appropriate device (GPU/CPU)
            x_xwav, x_ywav, x_zwav = x_xwav.to(self.device), x_ywav.to(self.device), x_zwav.to(self.device)
            x_xmel, x_ymel, x_zmel = x_xmel.to(self.device), x_ymel.to(self.device), x_zmel.to(self.device)

            # Perform inference (no gradient computation)
            with torch.no_grad():
                predict_ids, _ = net(x_xwav, x_ywav, x_zwav, x_xmel, x_ymel, x_zmel, None)
                predicted_label = torch.argmax(predict_ids, dim=1).cpu().item()  # Get the predicted label (0 or 1)
            
            # Use index as chunk_id (or can extract from file names)
            chunk_id = idx
            
            # Append prediction to the output list
            csv_output.append([chunk_id, predicted_label])

        # Save predictions to a CSV file
        if save:
            csv_path = os.path.join(result_dir, 'test_predictionscopy.csv')
            with open(csv_path, 'w') as f:
                f.write("chunk_id,label\n")  # Write header
                for chunk_id, label in csv_output:
                    f.write(f"{chunk_id},{label}\n")

            # Log the location of the saved results
            self.logger.info(f"Test results saved to {csv_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Test Script")
    # Load parameters from config file
    params = utils.load_yaml(file_path='./config.yaml')
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=utils.set_type)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()  # Parse arguments
    tester = Tester(args)  # Initialize Tester object
    tester.test(save=True)  # Run testing and save results
