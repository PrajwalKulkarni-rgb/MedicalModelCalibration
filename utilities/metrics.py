import torch
import os
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

class CalibrationMetrics:
    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def _compute_bin_boundaries(self, probabilities=None):
        if probabilities is None or probabilities.numel() == 0:  # Handle empty input safely
            return torch.linspace(0, 1, self.n_bins + 1, device="cuda")  # Default to cuda
        probabilities_sort = torch.sort(probabilities)[0]  # Sort tensor
        bin_n = len(probabilities) // self.n_bins
        bin_boundaries = torch.tensor([probabilities_sort[i * bin_n] for i in range(self.n_bins)], device=probabilities.device)
        return torch.cat([bin_boundaries, torch.tensor([1.0], device=probabilities.device)])


    def _get_probabilities(self, outputs, labels, logits=True):
        probabilities = F.softmax(outputs, dim=1) if logits else outputs
        confidences, predictions = torch.max(probabilities, dim=1)
        accuracies = predictions.eq(labels).float()
        return probabilities, confidences, predictions, accuracies

    def expected_calibration_error(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = self._compute_bin_boundaries()
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        ece = torch.tensor(0.0, device=outputs.device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * torch.abs(confidence_in_bin - accuracy_in_bin)
        return ece.item()

    def max_calibration_error(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = self._compute_bin_boundaries()
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        bin_scores = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                bin_scores.append(torch.abs(confidence_in_bin - accuracy_in_bin))
        return max(bin_scores).item() if bin_scores else 0.0

    def static_calibration_error(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=outputs.device)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        sce = torch.tensor(0.0, device=outputs.device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                sce += torch.abs(confidence_in_bin - accuracy_in_bin)
        return sce.item()

    def adaptive_calibration_error(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = self._compute_bin_boundaries(confidences)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        ace = torch.tensor(0.0, device=outputs.device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                confidence_in_bin = confidences[in_bin].mean()
                ace += torch.abs(confidence_in_bin - accuracy_in_bin)
        return ace.item()

    def compute_auc(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)

    # Convert labels to one-hot encoding
        labels_one_hot = F.one_hot(labels, num_classes=probabilities.shape[1]).float()

    # Check if every class has at least one positive sample
        valid_classes = labels_one_hot.sum(dim=0) > 0  # Boolean mask for valid classes

        if valid_classes.sum() < 2:  # If fewer than 2 classes are present in this batch
           print("Warning: Not enough class diversity in batch for AUC. Skipping computation.")
           return torch.tensor(0.0, device=outputs.device)  # Return zero instead of crashing

    # Compute AUC only for valid classes
        auc_score = roc_auc_score(
        labels_one_hot[:, valid_classes].cpu().numpy(),
        probabilities[:, valid_classes].cpu().numpy(),
        multi_class='ovr',
        average='weighted'
        )

        return torch.tensor(auc_score, device=outputs.device)



    def compute_f1(self, outputs, labels, logits=True):
        probabilities,confidences,predictions,accuracies = self._get_probabilities(outputs, labels, logits)
        return f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')

    
    def plot_reliability_diagram(self, outputs, labels, logits=True):
        probabilities, confidences, predictions, accuracies = self._get_probabilities(outputs, labels, logits)
        bin_boundaries = self._compute_bin_boundaries()
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        bin_accuracies = []
        bin_confidences = []
     
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin > 0:
               bin_accuracies.append(accuracies[in_bin].mean().cpu().item())
               bin_confidences.append(confidences[in_bin].mean().cpu().item())

        fig, ax = plt.subplots(figsize=(6, 6))  # Create figure and axis
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        ax.scatter(bin_confidences, bin_accuracies, color='red', label='Model Reliability')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.legend()

        return fig  # Return the figure object



