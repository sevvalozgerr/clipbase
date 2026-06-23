#!/usr/bin/env bash
set -e
python -m engine.train    --config config.yaml --routing none
python -m engine.evaluate --config config.yaml --routing none
