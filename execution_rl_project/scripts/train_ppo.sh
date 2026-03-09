#!/usr/bin/env bash
set -e
python -m src.agents.ppo_train
python -m src.agents.evaluate