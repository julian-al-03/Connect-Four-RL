import numpy as np
import torch

from connect4.game import ConnectFour
from connect4.mcts import MCTS, visit_counts
from connect4.networks import MLPNet, NetworkEvaluator, alphazero_loss


def encode_batch(states):
    return torch.from_numpy(np.stack([s.encode() for s in states]))


def play_moves(moves):
    state = ConnectFour()
    for col in moves:
        state = state.play(col)
    return state


def test_output_shapes_and_ranges():
    net = MLPNet()
    x = encode_batch([ConnectFour(), play_moves([3, 3, 4])])
    logits, value = net(x)
    assert logits.shape == (2, 7) and value.shape == (2,)
    assert torch.all(value.abs() <= 1)


def test_illegal_moves_get_zero_probability():
    net = MLPNet()
    logits, _ = net(encode_batch([play_moves([0] * 6)]))  # column 0 is full
    probs = torch.softmax(logits, dim=-1)[0]
    assert probs[0] == 0
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_can_overfit_small_batch():
    # Standard sanity check: the network and loss must be able to memorise a few examples
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    states = []
    for _ in range(16):
        state = ConnectFour()
        for _ in range(rng.integers(0, 10)):
            state = state.play(rng.choice(np.flatnonzero(state.legal_moves())))
        states.append(state)
    x = encode_batch(states)
    # random target distributions over the legal moves
    legal = torch.tensor(np.stack([s.legal_moves() for s in states]))
    target_policy = torch.softmax(torch.randn(16, 7), dim=-1) * legal
    target_policy /= target_policy.sum(dim=-1, keepdim=True)
    target_value = torch.tensor(rng.choice([-1.0, 0.0, 1.0], size=16), dtype=torch.float32)

    net = MLPNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(500):
        logits, value = net(x)
        loss, policy_loss, value_loss = alphazero_loss(logits, value, target_policy, target_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # cross-entropy cannot go below the entropy of the targets, so compare to that
    entropy = -(target_policy * torch.log(target_policy.clamp_min(1e-12))).sum(dim=-1).mean()
    assert policy_loss.item() - entropy.item() < 0.01
    assert value_loss.item() < 0.01


def test_network_evaluator_output():
    priors, value = NetworkEvaluator(MLPNet())(play_moves([0] * 6))
    assert priors.shape == (7,) and priors[0] == 0
    assert np.isclose(priors.sum(), 1.0)
    assert isinstance(value, float) and -1 <= value <= 1


def test_mcts_runs_with_network():
    root = MCTS(NetworkEvaluator(MLPNet()), num_simulations=50).search(ConnectFour(), add_noise=True)
    assert visit_counts(root).sum() == 49
