import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.optim as optim
import torch.nn as nn
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from utilities.misc import mkdir_p 
from utilities.__init__ import save_checkpoint
from argparsor import parse_args 
from utilities.data_loader import get_data_loaders, get_dataset_info
from models.resnet18 import ResNet18
from utilities.metrics import CalibrationMetrics  # Updated to use CalibrationMetrics

if __name__ == "__main__":
    args = parse_args()
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    dataset_info = get_dataset_info(args.dataset)
    dataset_names = [info['name'] for info in dataset_info]
    
    if args.dataset_name not in dataset_names:
        raise ValueError(f"Dataset {args.dataset_name} not found in {args.dataset}. Available datasets: {dataset_names}")
    
    dataset_index = dataset_names.index(args.dataset_name)
    
    train_loaders, val_loaders, test_loaders = get_data_loaders(args.dataset, args.batch_size)
    trainloader = train_loaders[dataset_index]
    valloader = val_loaders[dataset_index]
    testloader = test_loaders[dataset_index]
    
    model_save_pth = f"{args.checkpoint}/{args.dataset_name}/posthoc_{current_time}"
    mkdir_p(model_save_pth)
    
    log_file = os.path.join(model_save_pth, "calibration.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Applying post-hoc calibration on dataset: {args.dataset_name}")
    
    model = ResNet18(num_classes=args.num_classes).cuda()
    checkpoint = torch.load("A:/Model_Calibration/Code_Medical/checkpoints/Skills_evaluation_dataset/FL+MDCA_2025-02-11_18-22-42/model_best.pth")
    model.load_state_dict(checkpoint['state_dict'])
    logging.info("Model weights loaded successfully!")

    class CalibrationScaling(nn.Module):
        def __init__(self, model, num_classes):
            super(CalibrationScaling, self).__init__()
            self.base_model = model
            self.num_classes = num_classes
            self._freeze_model()

        def _freeze_model(self):
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

    class TemperatureScaling(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.T = nn.Parameter(torch.ones(1))  # Trainable temperature

            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

        def forward(self, x):
            logits = self.base_model(x)
            return logits / self.T

        def calibrate(self, val_loader):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.LBFGS([self.T], lr=0.01)

            def eval_loss():
                loss = 0.0
                for images, targets in val_loader:
                    images, targets = images.cuda(), targets.cuda()
                    outputs = self.forward(images)
                    loss += criterion(outputs, targets)
                return loss

            def closure():
                optimizer.zero_grad()
                loss = eval_loss()
                loss.backward()
                return loss

            optimizer.step(closure)
            
            logging.info(f"Optimal Temperature: {self.T.item()}")
            return self.T.item()

    class DirichletScaling(CalibrationScaling):
        def __init__(self, model, num_classes, Lambda=0.0, Mu=0.0):
            super().__init__(model, num_classes)
            self.scaling_layer = nn.Linear(num_classes, num_classes)
            self.Lambda = Lambda  # Regularization for weights
            self.Mu = Mu  # Regularization for bias

        def forward(self, x):
            logits = self.base_model(x)
            scaled_logits = torch.log_softmax(self.scaling_layer(logits), dim=1)
            return scaled_logits

        def regularizer(self):
            k = self.num_classes
            W, b = self.scaling_layer.parameters()

            w_loss = ((W**2).sum() - (torch.diagonal(W, 0)**2).sum()) / (k * (k - 1))
            b_loss = ((b**2).sum()) / k

            return self.Lambda * w_loss + self.Mu * b_loss

        def loss_func(self, outputs, targets):
            criterion = nn.CrossEntropyLoss()
            return criterion(outputs, targets) + self.regularizer()

        def calibrate(self, train_loader, val_loader, epochs=50, lr=0.001):
            optimizer = optim.Adam(self.scaling_layer.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                    optimizer.zero_grad()
                    scaled_outputs = self.forward(inputs)
                    loss = self.loss_func(scaled_outputs, targets)
                    loss.backward()
                    optimizer.step()

                if val_loader:
                    self.base_model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs, targets = inputs.cuda(), targets.cuda()
                            scaled_outputs = self.forward(inputs)
                            val_loss += criterion(scaled_outputs, targets)
                    logging.info(f'Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}')

            return self.scaling_layer

    def apply_posthoc_calibration(model, num_classes, train_loader, val_loader, args):
        if args.calibration == "temperature":
            logging.info("Applying Temperature Scaling...")
            calibrator = TemperatureScaling(model).cuda()
            optimal_temp = calibrator.calibrate(val_loader)
            logging.info(f"Optimal Temperature: {optimal_temp}")
            return calibrator
        
        elif args.calibration == "dirichlet":
            logging.info("Applying Dirichlet Scaling...")
            calibrator = DirichletScaling(model, num_classes, Lambda=args.Lambda, Mu=args.Mu).cuda()
            calibrator.calibrate(train_loader, val_loader)
            logging.info("Dirichlet Scaling completed.")
            return calibrator
        
        else:
            logging.info("No calibration applied. Returning original model.")
            return model

    calibrated_model = apply_posthoc_calibration(model, args.num_classes, trainloader, valloader, args)

    calibration_metrics = CalibrationMetrics()

    outputs, labels = [], []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = calibrated_model(inputs)
            outputs.append(logits)
            labels.append(targets)

    outputs = torch.cat(outputs).to('cuda')  # Ensure outputs are on CUDA
    labels = torch.cat(labels).to('cuda')  # Ensure labels are on CUDA


    test_metrics = {
        "accuracy": 100 * (outputs.argmax(dim=1) == labels).float().mean().item(),
        "ece": calibration_metrics.expected_calibration_error(outputs, labels),
        "mce": calibration_metrics.max_calibration_error(outputs, labels),
        "ace": calibration_metrics.adaptive_calibration_error(outputs, labels),
        "sce": calibration_metrics.static_calibration_error(outputs, labels),
        "auc": calibration_metrics.compute_auc(outputs, labels),
        "f1_score": calibration_metrics.compute_f1(outputs, labels)
    }

    logging.info("Post-hoc calibration completed.")
    logging.info(f"Test Metrics after Calibration: {test_metrics}")

    save_checkpoint({'state_dict': calibrated_model.state_dict()}, is_best=False, checkpoint=model_save_pth)

    logging.info("Generating Reliability Diagram...")
    fig = calibration_metrics.plot_reliability_diagram(outputs, labels)
    fig.savefig(os.path.join(model_save_pth, "reliability_diagram.png"))
