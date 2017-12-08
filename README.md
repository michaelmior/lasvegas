# Las Vegas AI

This repo implements an AI for the dice game [Las Vegas](https://boardgamegeek.com/boardgame/117959/las-vegas) based on [deep-Q learning with experience replay](https://arxiv.org/abs/1312.5602).
Currently this has only been tested with Python 3.6 using the Theano backend of Keras.

## Installation

    export PYTHONPATH=.
    pip install -r requirements.txt

## Getting started

The repository contains two scripts [`bin/lasvegas-train`](bin/lasvegas-train) and [`bin/lasvegas-eval`](bin/lasvegas-eval).
Both can be run with `--help` to list the available options.
By default running `lasvegas-train` will train a neural network in a game against random players and save the model to `model.h5`.
Models can be evaluated by running `lasvegas-eval -m model.h5` to produce the average score in games against random players.
