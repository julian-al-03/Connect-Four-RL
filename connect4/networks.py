"""Policy-value networks: f(s) = (policy logits, value).

Input: encoded board of shape (batch, 2, 6, 7), see ConnectFour.encode().
Output: policy logits of shape (batch, 7), with illegal columns masked out,
        and value of shape (batch,), in [-1, 1], for the player to move.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from connect4.game import COLS, ROWS

ILLEGAL_LOGIT = -1e9  # not -inf: 0 * -inf would give NaN in the cross-entropy


def legal_mask(x):
    """A column is legal if its top cell is empty in both planes. Shape (batch, COLS)."""
    return x[:, :, 0, :].sum(dim=1) == 0

class PolicyValueNet(nn.Module):
    """Shared logic: subclasses define `body` and the two heads; this class masks the policy."""

    def forward(self, x):
        features = self.body(x)
        logits = self.policy_head(features)
        logits = logits.masked_fill(~legal_mask(x), ILLEGAL_LOGIT)
        value = torch.tanh(self.value_head(features)).squeeze(-1)
        return logits, value


class MLPNet(PolicyValueNet):
    """Fully connected network on the flattened board (2 * 6 * 7 = 84 inputs)."""

    def __init__(self, hidden=256, num_layers=3):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = 2 * ROWS * COLS
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, hidden), nn.ReLU()]
            in_features = hidden
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden, COLS)
        self.value_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))


def alphazero_loss(logits, value, target_policy, target_value):
    """(z - v)^2 - pi^T log p, averaged over the batch.

    The L2 term of the paper is handled by the optimizer's weight decay.
    Returns (total, policy_loss, value_loss) so both parts can be logged separately.
    """
    policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, target_value)
    return policy_loss + value_loss, policy_loss, value_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class NetworkEvaluator:
    """Leaf evaluator for MCTS: state -> (priors, value) from a policy-value network.

    Drop-in replacement for RolloutEvaluator: the search does not know which one it uses.
    """

    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, state):
        self.model.eval()  # set every call: training switches the model back to train mode
        x = torch.from_numpy(state.encode()).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        priors = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return priors, value.item()
