# Connect Four with AlphaZero-style Reinforcement Learning

Training a Connect Four agent from self-play only, following the AlphaZero approach
(Silver et al., 2017; 2018). An MLP policy-value network is compared against a
residual CNN within the same self-play pipeline.

## Setup

Requires Python 3.10 (the version used on the Gorina4 GPU cluster).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project structure

_Work in progress._

## Reproducing the experiments

The experiments ran on CPU only, on a machine with 48 threads. On a smaller machine, lower
`--workers` and `--train-threads`, or run fewer jobs at once. Results may differ slightly
across machines and library versions.

Run all commands in `bash` from the repository root:

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
100 simulations and then as the raw policy without search (`:0`).

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

### 7. Figures

```bash
python -m connect4.plot "mlp=runs/main/mlp_seed*" "cnn=runs/main/cnn_seed*" --out figures/main
python -m connect4.plot "temp 0=runs/sweep/mlp_temp0_seed?" "temp 5=runs/sweep/mlp_temp5_seed?" \
  "temp 10=runs/main/mlp_seed?" "temp 20=runs/sweep/mlp_temp20_seed?" --out figures/temperature
python -m connect4.plot "c_puct 0.5=runs/sweep/mlp_cpuct0.5_seed?" "c_puct 1.5=runs/main/mlp_seed?" \
  "c_puct 4=runs/sweep/mlp_cpuct4_seed?" --out figures/cpuct
```
