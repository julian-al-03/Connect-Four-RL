# Connect Four with AlphaZero-style Reinforcement Learning

Training a Connect Four agent from self-play only, following the AlphaZero approach
(Silver et al., 2017; 2018). An MLP policy-value network is compared against a
residual CNN within the same self-play pipeline.

## Setup

Requires Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

```
connect4/              the package; every script is run as python -m connect4.<name>
  game.py              lightweight game state (board, legal moves, win check) used by the search
  mcts.py              Monte Carlo tree search with PUCT selection and Dirichlet root noise
  networks.py          MLP and residual CNN policy-value networks, and the AlphaZero loss
  selfplay.py          self-play games that record training samples (s, pi, z), and mirroring
  train.py             training loop: self-play -> replay buffer -> gradient steps -> evaluation
  agents.py            baselines: random, greedy (one-move lookahead) and pure MCTS
  arena.py             plays games between agents, with random openings and both colours
  evaluate.py          final evaluation of agents against each other or against baselines
  parallel.py          process pool shared by self-play and evaluation
  plot.py              learning curves (scores, losses, score by time) from training logs
  plot_progress.py     score of each checkpoint against the previous one
  play.py              play against an agent in the terminal
tests/                 unit tests for every module (run with python -m pytest)
notebooks/             early exploration of the PettingZoo environment and the game class
```

Training writes each run to its own folder under `runs/` (not tracked by git), containing
`config.json` (all settings), `log.csv` (losses and scores per iteration) and a checkpoint
every `--eval-every` iterations. Figures are written to `figures/`.

Agents are written as strings, both in `evaluate.py` and `play.py`: `random`, `greedy`,
`pure:SIMS`, `NET:CHECKPOINT` (network + MCTS) and `NET:CHECKPOINT:0` (network without
search), where `NET` is `mlp` or `cnn`. For example, to play against a trained CNN:

```bash
python -m connect4.play cnn:runs/main/cnn_seed0/model_100.pt --simulations 100
```

## Reproducing the experiments

Run all commands from the repository root:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
mkdir -p runs/main runs/sweep
```

Every other hyperparameter uses the default in `connect4/train.py`. Each run folder contains
`config.json` (the exact settings), `log.csv` (losses and scores per iteration) and the
checkpoints.

### 1. MLP vs CNN

Trains both networks with 3 seeds each. An MLP run took about 1.2 hours and a CNN run about
4 hours.

```bash
for net in mlp cnn; do for seed in 0 1 2; do
  python -m connect4.train --net $net --seed $seed --iterations 100 --games-per-iteration 100 \
    --simulations 100 --train-steps 400 --buffer-size 100000 --eval-every 10 --eval-games 40 \
    --workers 6 --train-threads 2 --out runs/main/${net}_seed$seed \
    > runs/main_${net}_seed$seed.log 2>&1 &
done; done; wait
```

### 2. Final evaluation of the main runs

Plays the final networks against each other and against the baselines, first with MCTS at
100 simulations and then as the raw policy without search.

```bash
MLP=$(for f in runs/main/mlp_seed*/model_100.pt; do echo "mlp:$f"; done)
CNN=$(for f in runs/main/cnn_seed*/model_100.pt; do echo "cnn:$f"; done)
MLP_RAW=$(for a in $MLP; do echo "$a:0"; done)
CNN_RAW=$(for a in $CNN; do echo "$a:0"; done)
BASELINES="random greedy pure:100 pure:400"

python -m connect4.evaluate --agents $CNN --opponents $MLP --games 200 --simulations 100 \
  --workers 40 --out runs/main/final_head_to_head.csv
python -m connect4.evaluate --agents $MLP $CNN --opponents $BASELINES --games 200 --simulations 100 \
  --workers 40 --out runs/main/final_baselines.csv
python -m connect4.evaluate --agents $CNN_RAW --opponents $MLP_RAW --games 200 \
  --workers 40 --out runs/main/final_raw_head_to_head.csv
python -m connect4.evaluate --agents $MLP_RAW $CNN_RAW --opponents $BASELINES --games 200 \
  --workers 40 --out runs/main/final_raw_baselines.csv
```

### 3. Progress within each run

Plays iterations 20, 50 and 100 of each run against each other.

```bash
for net in mlp cnn; do for d in runs/main/${net}_seed*; do
  python -m connect4.evaluate --agents $net:$d/model_020.pt $net:$d/model_050.pt $net:$d/model_100.pt \
    --games 200 --simulations 100 --workers 40 --out $d/progress.csv
done; done
```

Plays each checkpoint against the one 10 iterations earlier (20 vs 10, ..., 100 vs 90).
Finished pairings are skipped, so the loop can be restarted.

```bash
for net in mlp cnn; do for d in runs/main/${net}_seed?; do
  for later in 20 30 40 50 60 70 80 90 100; do
    earlier=$((later - 10)); out=$d/consecutive_$later.csv
    [ -f "$out" ] || python -m connect4.evaluate --agents $net:$d/model_$(printf %03d $later).pt \
      --opponents $net:$d/model_$(printf %03d $earlier).pt --games 200 --simulations 100 \
      --workers 16 --seed $later --out "$out"
  done
done; done
```

### 4. Simulation sweep (MLP)

The main MLP runs provide the reference at 100 simulations. All final evaluations use 100
simulations, whatever the network was trained with.

```bash
for s in 25 50 200; do for seed in 0 1 2; do
  python -m connect4.train --net mlp --iterations 100 --games-per-iteration 100 --simulations $s \
    --train-steps 400 --buffer-size 100000 --eval-every 10 --eval-games 40 --seed $seed \
    --workers 5 --train-threads 2 --out runs/sweep/mlp_sims${s}_seed$seed \
    > runs/sweep/mlp_sims${s}_seed$seed.log 2>&1 &
done; done; wait

AGENTS=$(for f in runs/sweep/mlp_sims*/model_100.pt; do echo "mlp:$f"; done)
python -m connect4.evaluate --agents $AGENTS --opponents pure:100 pure:400 --games 200 \
  --simulations 100 --workers 40 --out runs/sweep/final_sweep.csv
```

### 5. Temperature sweep (MLP)

The main MLP runs provide the reference at `--temperature-moves 10`.

```bash
for t in 0 5 20; do for seed in 0 1 2; do
  python -m connect4.train --net mlp --iterations 100 --games-per-iteration 100 --simulations 100 \
    --temperature-moves $t --train-steps 400 --buffer-size 100000 --eval-every 10 --eval-games 40 \
    --seed $seed --workers 5 --train-threads 2 --out runs/sweep/mlp_temp${t}_seed$seed \
    > runs/sweep/mlp_temp${t}_seed$seed.log 2>&1 &
done; done; wait

AGENTS=$(for f in runs/sweep/mlp_temp*/model_100.pt; do echo "mlp:$f"; done)
python -m connect4.evaluate --agents $AGENTS --opponents pure:100 pure:400 --games 200 \
  --simulations 100 --workers 40 --out runs/sweep/final_temperature.csv
```

### 6. `c_puct` sweep (MLP)

The main MLP runs provide the reference at `--c-puct 1.5`. The final evaluation uses
`c_puct` 1.5 for all agents, whatever value the network was trained with.

```bash
for c in 0.5 4; do for seed in 0 1 2; do
  python -m connect4.train --net mlp --iterations 100 --games-per-iteration 100 --simulations 100 \
    --c-puct $c --train-steps 400 --buffer-size 100000 --eval-every 10 --eval-games 40 \
    --seed $seed --workers 6 --train-threads 2 --out runs/sweep/mlp_cpuct${c}_seed$seed \
    > runs/sweep/mlp_cpuct${c}_seed$seed.log 2>&1 &
done; done; wait

AGENTS=$(for f in runs/sweep/mlp_cpuct*/model_100.pt; do echo "mlp:$f"; done)
python -m connect4.evaluate --agents $AGENTS --opponents pure:100 pure:400 --games 200 \
  --simulations 100 --workers 40 --out runs/sweep/final_cpuct.csv
```

### 7. CNN sweeps

Repeats sections 4 to 6 for the CNN, with the main CNN runs as the reference. Six runs share
the machine at a time; finished runs are skipped, so the command can be restarted. This took
about a day.

```bash
mkdir -p runs/sweep_cnn

while read -r name flags; do
  for seed in 0 1 2; do echo "$name $seed $flags"; done
done <<'EXPERIMENTS' | xargs -P 6 -L 1 bash -c '
  name=$1; seed=$2; shift 2
  out=runs/sweep_cnn/cnn_${name}_seed$seed
  [ -f "$out/model_100.pt" ] && exit 0
  python -m connect4.train --net cnn --seed "$seed" --iterations 100 --games-per-iteration 100 \
    --simulations 100 --train-steps 400 --buffer-size 100000 --eval-every 10 --eval-games 40 \
    --workers 6 --train-threads 2 "$@" --out "$out" > "$out.log" 2>&1
' _
temp20 --temperature-moves 20
temp0 --temperature-moves 0
cpuct4 --c-puct 4
cpuct0.5 --c-puct 0.5
temp5 --temperature-moves 5
sims200 --simulations 200
sims50 --simulations 50
sims25 --simulations 25
EXPERIMENTS

for group in temp cpuct sims; do
  AGENTS=$(for f in runs/sweep_cnn/cnn_${group}*/model_100.pt; do echo "cnn:$f"; done)
  python -m connect4.evaluate --agents $AGENTS --opponents pure:100 pure:400 --games 200 \
    --simulations 100 --workers 40 --out runs/sweep_cnn/final_$group.csv
done
```

### 8. Training only on new games (MLP)

Trains the MLP with 500 games per iteration and `--fresh-buffer`, which empties the buffer at
the start of every iteration, so each iteration trains only on its own games. The large
`--buffer-size` is only an upper limit, so no new game is cut. The main MLP runs provide the
reference.

```bash
mkdir -p runs/fresh
for seed in 0 1 2; do
  python -m connect4.train --net mlp --seed $seed --iterations 100 --games-per-iteration 500 \
    --simulations 100 --train-steps 400 --buffer-size 1000000 --fresh-buffer --eval-every 10 \
    --eval-games 40 --workers 14 --train-threads 2 --out runs/fresh/mlp_games500_seed$seed \
    > runs/fresh/mlp_games500_seed$seed.log 2>&1 &
done; wait

FRESH=$(for f in runs/fresh/mlp_games500_seed?/model_100.pt; do echo "mlp:$f"; done)
python -m connect4.evaluate --agents $FRESH --opponents pure:100 greedy --games 200 \
  --simulations 100 --workers 40 --out runs/fresh/final_baselines.csv
```

### 9. Figures

```bash
python -m connect4.plot "mlp=runs/main/mlp_seed*" "cnn=runs/main/cnn_seed*" --out figures/main
python -m connect4.plot_progress "mlp=runs/main/mlp_seed?" "cnn=runs/main/cnn_seed?" --out figures/main
python -m connect4.plot "temp 0=runs/sweep/mlp_temp0_seed?" "temp 5=runs/sweep/mlp_temp5_seed?" \
  "temp 10=runs/main/mlp_seed?" "temp 20=runs/sweep/mlp_temp20_seed?" --out figures/temperature
python -m connect4.plot "c_puct 0.5=runs/sweep/mlp_cpuct0.5_seed?" "c_puct 1.5=runs/main/mlp_seed?" \
  "c_puct 4=runs/sweep/mlp_cpuct4_seed?" --out figures/cpuct
python -m connect4.plot "main=runs/main/mlp_seed?" "fresh=runs/fresh/mlp_games500_seed?" --out figures/fresh
```
