import random

from tqdm import tqdm

from lasvegas import constants, turns
from lasvegas.state import State


def eval_model(agent, games, baseline, model=None, baseline_model=None):
    # Set up the turn functions
    agent = getattr(turns, agent + '_turn')
    if baseline != 'changing':
        baseline = [getattr(turns, baseline + '_turn')] * \
                   (constants.MAX_PLAYERS - 1)

    total = 0
    wins = 0
    draws = 0
    losses = 0
    for game in tqdm(range(games)):
        s = State.initial()

        # Pick a new strategy for each player
        if baseline == 'changing':
            baseline = [getattr(turns, random.choice([
                'biggest',
                'greedy',
                'random',
                'richest'
            ]) + '_turn') for _ in range(constants.MAX_PLAYERS - 1)]

        while not s.game_end():
            while not s.round_end():
                # Run the agent we're evaluating
                s = agent(s, 0, model)

                # Run turns for all other agents
                for i in range(1, s.players):
                    s = baseline[i - 1](s, i, baseline_model)
            if s.round_num != constants.MAX_ROUNDS:
                s = s.advance_round()
        total += s.cash[0]

        max_cash = max(s.cash)
        if s.cash[0] != max_cash:
            losses += 1
        elif s.cash.count(max_cash) == 1:
            wins += 1
        else:
            draws += 1

    return {'total': total, 'wins': wins, 'draws': draws, 'losses': losses}
