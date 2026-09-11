import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional

from models.causalfall import CausalFall
from training.loss import CausalFallLoss
from evaluation.metrics import compute_metrics

class Trainer:
    """
    Trainer for CausalFall and baseline models.
    [PAPER SPECIFICATION] Section 5.2.3:
        - Optimizer: Adam
        - Learning Rate: 1e-3
        - Batch Size: 128
        - Epochs: 500
        - Loss: L_Global = L_Decoder + L_Counter
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        device: str = "auto",
        checkpoint_dir: str = "./checkpoints",
        is_causalfall: bool = True
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.is_causalfall = is_causalfall
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # [PAPER SPECIFICATION] Optimizer: Adam with lr=1e-3
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = CausalFallLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_dec = 0.0
        total_count = 0.0
        n_batches = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.is_causalfall:
                outputs = self.model(batch_x)
                loss_global, loss_dec, loss_count = self.loss_fn(
                    outputs['y_o'],
                    outputs['y_causal'],
                    outputs['y_cf'],
                    batch_y
                )
                loss = loss_global
                total_dec += loss_dec.item()
                total_count += loss_count.item()
            else:
                # Baseline model output (logits or probs)
                logits = self.model(batch_x)
                loss = self.ce_loss(logits, batch_y)
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        return {
            'loss_global': total_loss / max(1, n_batches),
            'loss_decoder': total_dec / max(1, n_batches),
            'loss_counter': total_count / max(1, n_batches)
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(self.device)
            
            if self.is_causalfall:
                outputs = self.model(batch_x)
                # Debiased prediction Y_Causal used for classification
                probs = outputs['y_causal'].cpu().numpy()
                preds = np.argmax(probs, axis=-1)
            else:
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        return metrics

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        history = []
        
        for epoch in range(1, self.epochs + 1):
            train_losses = self.train_epoch()
            eval_metrics = self.evaluate()
            
            history.append({
                'epoch': epoch,
                **train_losses,
                **eval_metrics
            })
            
            if eval_metrics['F1-Score'] > best_f1:
                best_f1 = eval_metrics['F1-Score']
                best_metrics = eval_metrics
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pt"))
                
            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == self.epochs):
                print(
                    f"Epoch [{epoch:3d}/{self.epochs}] "
                    f"Loss: {train_losses['loss_global']:.4f} | "
                    f"Acc: {eval_metrics['Accuracy']:.2f}% | "
                    f"Prec: {eval_metrics['Precision']:.2f}% | "
                    f"Rec: {eval_metrics['Recall']:.2f}% | "
                    f"F1: {eval_metrics['F1-Score']:.2f}%"
                )
                
        return {
            'best_metrics': best_metrics,
            'final_metrics': eval_metrics,
            'history': history
        }
