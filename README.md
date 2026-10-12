# HMMSM: Bayesian Hidden Markov Model with Singh-Maddala Emissions

Reference implementation of the paper:

> Kadhem, S. K. (2026). *A Bayesian Hidden Markov Model with Singh-Maddala Emission Distributions for Heavy-Tailed Regime-Switching Time Series.*

## Overview

This repository provides the complete Python implementation of HMMSM, including:

- Full Bayesian inference via Metropolis-within-Gibbs with FFBS
- Simulation study scripts
- Application to 26 years of gridded Irish rainfall data
- MCMC convergence diagnostics (R-hat, ESS, Geweke)
- Reparameterization to (α, μ, q)

## Requirements

- Python 3.11+
- See `requirements.txt`

## Installation

```bash
git clone https://github.com/safaa-kadhem/hmmsm-singh-maddala.git
cd hmmsm-singh-maddala
pip install -r requirements.txt
