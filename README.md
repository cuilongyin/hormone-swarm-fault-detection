# Hormone-Based Fault Detection for Robot Swarms

This repository contains the simulation code for the paper:

**"Ratio-Based Signaling for Source-Victim Separation in Swarm Fault Detection"**  
Accepted at AAMAS 2026

## Overview

This code implements and evaluates a bio-inspired hormone-based fault detection method for robot swarms, along with baseline comparisons.

## Requirements

See `requirements.txt` for dependencies. Install with:
```
pip install -r requirements.txt
```

## Project Structure

- `src/core/`: Core simulation components (agents, metrics, simulator)
- `src/methods/`: Fault detection methods (hormone-based, threshold, voting, baseline)
- `experiments/`: Scripts for running experiments from the paper
- `run_all.py`: Main script to reproduce all results

## Running Experiments

To reproduce the results from the paper:
```
python run_all.py
```

Individual experiments can be run from the `experiments/` folder.

## Citation

If you use this code, please cite:
```
[Your citation will go here after camera-ready]
```

## Contact

[Your email] - feel free to reach out with questions
