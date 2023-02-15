import math
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from utils.nn_utils import MLP
from utils.stats import EPSILON


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, running_mean):
        ctx.save_for_backward(x, running_mean)
        input_log_sum_exp = x.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        cur, running_mean = ctx.saved_tensors
        grad = grad_output * cur.exp().detach() / \
            (running_mean + EPSILON) / cur.shape[0]
        return grad, None


class Mine(pl.LightningModule):
    def __init__(self, seed, data_dim, hidden_dims, include_y, alpha, lr):
        super().__init__()
        self.save_hyperparameters()
        self.seed = seed
        self.include_y = include_y
        self.alpha = alpha
        self.running_mean = 0
        self.lr = lr
        self.net = MLP(3 * data_dim + (1 if include_y else 0), hidden_dims, 1)


    def ema_loss(self, x):
        t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(len(x))).detach()
        if self.running_mean == 0:
            self.running_mean = t_exp
        else:
            self.running_mean = self.alpha * t_exp + (1 - self.alpha) * self.running_mean.item()
        factorized_term = EMALoss.apply(x, self.running_mean)
        return factorized_term


    def loss(self, u, x, y):
        args_left = u
        if self.include_y:
            args_right = torch.cat((x, y), dim=1)
        else:
            args_right = x
        args_right_perm = args_right[torch.randperm(len(args_right))]
        joint_term = -self.net(args_left, args_right).mean()
        factorized_term = self.ema_loss(self.net(args_left, args_right_perm))
        return joint_term + factorized_term


    def training_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        loss = self.loss(*batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)