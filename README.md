# Racing Neural Network

A small Python and Pygame experiment that evolves a neural network to drive a car around a 2D track. The project uses a genetic algorithm: networks drive simulated cars, receive fitness from track progress, and produce mutated offspring for the next generation.

The project supports:

- Training on multiple tracks
- Persistent best weights per track
- Checkpoint-based progress measurement
- Recovery trials from the furthest learned checkpoint
- Random immigrants and adaptive mutation when progress stalls
- A one-car test mode for evaluating saved weights
- Shared multi-track training with one population evaluated across every training track
- Evaluation on a track that was not used during training

## Requirements

- Python 3.10 or newer
- Pygame
- A desktop environment capable of opening a Pygame window

Install the dependency with:

```powershell
python -m pip install pygame
```

## Quick Start

From the project directory, start training on the included track:

```powershell
python game.py --mode train --track track1.png
```

The default track uses `checkpoints.json`. Training saves:

- `best_car.json`: best normal-start network for track 1
- `checkpoint_archive.json`: networks that reached individual checkpoints

The files are created and updated in the project directory. They are useful for continuing training, but generated weights are not required to run the source code from scratch.

## Test A Saved Network

Test the saved track 1 network from the starting line:

```powershell
python game.py --mode test --track track1.png --weights best_car.json
```

Test mode runs exactly one car. It stops when the car completes a lap, crashes, reaches the timeout, or the window is closed. The result is printed in the terminal.

To require multiple consecutive laps during a test:

```powershell
python game.py --mode test --track track1.png --weights best_car.json --laps 3
```

Training still uses one lap per generation by default. This is intentional: repeated laps make evaluation slower without giving the genetic algorithm a better first signal.

You can change the timeout in milliseconds:

```powershell
python game.py --mode test --track track1.png --weights best_car.json --timeout 60000
```

## Add Another Track

1. Add a new image such as `track2.png`.
2. Create an ordered checkpoint file for it:

```powershell
python checkpoint-picker.py --track track2.png --output track2_checkpoints.json
```

3. Click checkpoints in driving order. Press `Z` to undo the last point and `S` to save.
4. Train the new track:

```powershell
python game.py --mode train --track track2.png
```

When the track is named `track2.png`, the runner automatically uses:

- `track2_checkpoints.json`
- `best_track2.json`
- `track2_checkpoint_archive.json`

You can provide explicit paths when needed:

```powershell
python game.py --mode train `
    --track custom_track.png `
    --checkpoints custom_checkpoints.json `
    --weights custom_best.json `
    --archive custom_archive.json
```

PowerShell accepts the backtick character for line continuation. On one line, the same command is:

```powershell
python game.py --mode train --track custom_track.png --checkpoints custom_checkpoints.json --weights custom_best.json --archive custom_archive.json
```

## Unseen-Track Evaluation

To evaluate generalisation, do not train on the final track. Add its image and checkpoints, then test weights produced from a different track:

```powershell
python game.py --mode test `
    --track final_unseen_track.png `
    --checkpoints final_unseen_track_checkpoints.json `
    --weights best_car.json
```

The neural network only receives five local distance sensors, so it does not receive the track image or checkpoint coordinates directly. Checkpoints are used by the evolutionary fitness system and are not network inputs. For a meaningful unseen-track test, keep the final track and its checkpoint file out of the training process.

## Shared Multi-Track Training

Train one shared model across the known tracks with:

```powershell
python game.py --mode multi-train --tracks track1.png,track2.png,track3.png --checkpoint-files track1_checkpoints.json,track2_checkpoints.json,track3_checkpoints.json --weights shared_best.json
```

The simulations update every frame, but the dashboard is rendered every three frames by default to keep visualisation from slowing training. For more speed, render less often:

```powershell
python game.py --mode multi-train --tracks track1.png,track2.png,track3.png --checkpoint-files track1_checkpoints.json,track2_checkpoints.json,track3_checkpoints.json --weights shared_best_v2.json --render-every 10
```

Use `--render-every 1` for a dashboard update every frame. A larger value changes only how often the screen is drawn, not how often the cars are simulated.

The same population is evaluated on every listed track in each generation. Each genome receives a normalized score per track, then a combined score:

```text
combined score = 0.3 * average track score + 0.7 * worst track score
```

The worst-track term prevents a genome from succeeding on two tracks while failing the third. The lowest-scoring track is recalculated for every genome and every generation, so no track is permanently privileged. The mode seeds the new population from existing per-track champions when available, while retaining random genomes and later immigrants. Every car starts from the real start line; checkpoint recovery starts are disabled so the shared score remains fair.

The selected champion's individual track scores and current hardest track are printed each generation. Save the shared model as `shared_best.json`, and never include the final unseen track in either the `--tracks` or `--checkpoint-files` lists.

During shared training, the fourth dashboard panel is a neural observer for the current best Track 1 car. It visualizes the five sensor values, three hidden-neuron activations, steering and throttle outputs, and the resulting driving decision. The other panels continue to show the track simulations. This is diagnostic display only and does not change training or give the network extra inputs.

After five stagnant generations, the trainer launches a small recovery experiment on the currently weakest track. Recovery cars start near the furthest checkpoint reached by normal-start cars and explore mutated policies locally. Their recovery scores are never used directly to save a model or select the champion. A recovered policy is inserted into the next population and must then perform from the normal start on every track before it can win.

## Training Tutorial

### 1. Start clean

If you want a genuinely fresh experiment, move the generated files somewhere safe or remove them:

```powershell
Move-Item best_car.json best_car.previous.json -ErrorAction SilentlyContinue
Move-Item checkpoint_archive.json checkpoint_archive.previous.json -ErrorAction SilentlyContinue
```

Do not remove the track images, car sprite, or checkpoint files.

### 2. Train the first track

Train the easiest track first:

```powershell
python game.py --mode train --track track1.png
```

Leave training running until the normal-start model reaches every checkpoint and completes several reliable test runs:

```powershell
python game.py --mode test --track track1.png --weights best_car.json
```

Do not judge progress from one lucky lap. Repeat the test several times and record the checkpoint count, crash/stuck reason, and lap time.

### 3. Add two more training tracks

Create checkpoints in driving order:

```powershell
python checkpoint-picker.py --track track2.png --output track2_checkpoints.json
python checkpoint-picker.py --track track3.png --output track3_checkpoints.json
```

Train them separately at first:

```powershell
python game.py --mode train --track track2.png
python game.py --mode train --track track3.png
```

Each track gets its own weights and archive. This makes it possible to compare track difficulty without overwriting another track's model.

### 4. Keep one track unseen

Do not run training with the final evaluation track or its checkpoints. Test a model trained on another track against it only after the training tracks are stable:

```powershell
python game.py --mode test --track final_track.png --checkpoints final_track_checkpoints.json --weights best_car.json
```

### 5. Train the shared model

The shared trainer evaluates one population on every listed training track. It uses normalized scores and dynamic worst-track pressure, so whichever track is hardest at a given time receives the selection pressure automatically.

Keep the final evaluation track out of both track lists. Separate per-track models remain useful baselines, but `shared_best.json` is the model intended for generalisation.

## How Training Works

Each car uses a small feed-forward network:

- Five inputs from forward-facing distance sensors
- One hidden layer with three neurons
- Two outputs for steering and throttle

A generation ends when every car has crashed or completed a lap, or when the generation timeout is reached. Fitness rewards:

- Passing checkpoints
- Getting closer to the next checkpoint
- Surviving and moving away from the start
- Completing a lap, with shorter laps preferred

The next generation contains an elite normal-start car, mutated offspring, and random immigrants. After prolonged stagnation, mutation strength and mutation rate increase.

## Recovery Without Detraining

Recovery starts are deliberately isolated from the main training lineage:

- Normal-start cars decide which network is saved as the best model.
- Only normal-start cars are used as parents for the next generation.
- Recovery cars can extend the checkpoint archive if they reach a new checkpoint.
- A recovery car cannot replace an existing checkpoint specialist just because it started closer to the finish.
- The archived network is tested from its checkpoint as a local improvement experiment, not treated as proof that the full-start network improved.

This means checkpoint recovery can explore difficult sections without allowing a head-start score to overwrite the model that still has to complete the entire track from the beginning. The final authority remains test mode from the starting line.

## Circular-Motion Detection

Cars are also ended early when they appear to be looping in place. This is not based on speed alone, because a car may need to slow down for a corner. Every 2.5 seconds, the simulator checks whether the car:

- Travelled at least 120 pixels in total
- Finished within 40 pixels of where that observation began

That combination indicates substantial motion with almost no forward progress, which is a useful signal for spinning or oscillating. Cars that are merely moving slowly or taking a normal wide corner are not ended by the looping check. A separate three-second stationary timeout ends cars that stop moving entirely. A checkpoint reset also gives a recovery car a fresh observation window.

The thresholds are defined near the top of `utils.py` as `STUCK_CHECK_INTERVAL_MS`, `STUCK_MIN_TRAVEL_DISTANCE`, and `STUCK_MAX_NET_DISPLACEMENT`. If legitimate tight corners are being ended too early, increase the interval or reduce the allowed travel threshold. If cars visibly spin for too long, lower the interval or increase the allowed net displacement threshold.

## Checkpoint Guidelines

Place checkpoints:

- In the direction of travel
- On the drivable surface
- Near the centre of the road
- Far enough apart that a car cannot accidentally skip the intended route
- Before difficult corners or junctions when extra guidance is useful

Checkpoint order matters. A checkpoint file is simply an ordered JSON list of `[x, y]` coordinate pairs.

Cars do not require every track to have the same start-line orientation. On a track with checkpoints, each normal-start car automatically points from the detected start position toward the first checkpoint. Recovery cars point toward the next checkpoint in their recovery section. For a track without checkpoints, the simulator falls back to the default 270-degree starting angle, so add at least one checkpoint when the route direction matters.

### Checkpoint gates

Checkpoints are scored as directional gates rather than small target circles. The simulator draws an orange line through each checkpoint, perpendicular to the local route direction. A car passes a checkpoint when its movement crosses that line while travelling in the intended forward direction. Being near the orange circle is not enough.

The gate currently extends 55 pixels on either side of its checkpoint centre. This should cover the road in the included tracks. When creating a new track, inspect the drawn orange lines while training. If a line does not reach both road edges, adjust `CHECKPOINT_GATE_HALF_WIDTH` in `utils.py`. Place checkpoint centres near the road centre and keep them away from junctions where a single gate could be crossed accidentally.

The car may advance to a later gate if it crosses that gate first while moving in its intended direction. This supports a genuine shortcut: the skipped section is treated as completed, but the car still has to cross the later gates and finish the lap. The route geometry, gate width, and checkpoint placement therefore determine which shortcuts are valid; the neural network does not receive checkpoint positions as inputs.

## Project Files

- `game.py`: command-line training and testing runner
- `utils.py`: car physics, sensors, fitness, evolution, persistence, and recovery
- `network.py`: feed-forward neural network functions
- `checkpoint-picker.py`: interactive checkpoint authoring tool
- `track1.png`: included training track
- `checkpoints.json`: included track checkpoint coordinates
- `car.png`: car sprite
- `best_car.json`: generated best weights for the default track
- `checkpoint_archive.json`: generated checkpoint recovery archive

## Open Source Notes

This repository is intended to be experimented with and extended. Contributions could include better sensors, richer fitness functions, replay tools, additional tracks, configuration files, or automated evaluation.

Before publishing, choose and add a license file that matches how you want others to use the project. A `LICENSE` file is not included automatically because the desired license belongs to the project owner.
