# Hormone-Based Fault Detection for Robot Swarms

This repository contains the simulation code for the paper:

**"Ratio-Based Signaling for Source-Victim Separation in Swarm Fault Detection"**  
AAMAS 2026

## Overview

This code implements a bio-inspired hormone-based fault detection method for robot swarms and baseline comparisons used in our experiments.

## Requirements

See `requirements.txt` for dependencies. Install with:
```
pip install -r requirements.txt
```

## Project Structure

- `src/core/`: Core simulation components (agents, metrics, simulator)
- `src/methods/`: Fault detection methods (hormone-based, threshold, voting, baseline)
- `experiments/`: 7 Scripts for running experiments from the paper
- `run_all.py`: Main script to reproduce all results

## Running Experiments

To reproduce the results from the paper:
```
python run_all.py
```

Individual experiments can be run from the `experiments/` folder.

## Note on Code Quality

This code was developed for research purposes and is provided as-is to support the reproducibility of our published results. While we have tested the implementation, there may be (deep breath) ... opportunities for optimization and improvement. We welcome feedback and are actively maintaining this codebase.

## Citation

If you use this code, please cite our AAMAS 2026 paper:
```
Longyin Cui. 2026. Ratio-Based Signaling for Source-Victim Separation in 
Swarm Fault Detection. In Proc. of the 25th International Conference 
on Autonomous Agents and Multiagent Systems (AAMAS 2026)
```

## Contact

For questions or collaboration opportunities, please contact [your email]
