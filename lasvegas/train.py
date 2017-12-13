import copy
import random

import numpy as np
from tqdm import tqdm

from lasvegas import constants, turns
from lasvegas.state import roll, State


def train_model(model, games, baseline, baseline_model=None):
    """
    Train the given model for a number of games against a baseline
    """
    memory = []
    game_num = 0

    if baseline != 'changing':
        baseline = [getattr(turns, baseline + '_turn')] * \
                   (constants.MAX_PLAYERS - 1)

    for _ in tqdm(range(games)):
        state = State.initial()

        # Pick a new strategy for each player
        if baseline == 'changing':
            baseline = [getattr(turns, random.choice([
                'biggest',
                'greedy',
                'random',
                'richest'
            ]) + '_turn') for _ in range(constants.MAX_PLAYERS - 1)]

        while not state.game_end():
            # Roll the dice for player zero
            left = state.dice_left(0)
            r = roll(left)

            # Pick random action EPSILON% of the time
            if random.random() < constants.EPSILON:
                p = [random.random() for _ in range(6)]
            else:
                p = model.predict(np.array([state.as_vector(r)]))[0]

            # Check if the given action was valid and then
            # pick the best valid action if needed
            valid = r[max(enumerate(p), key=lambda x: x[1])[0]] > 0
            a = turns.choose_action(p, r)

            # Make a copy of the state with the updated dice
            new_state = copy.deepcopy(state)
            new_state.spots[a].dice[0] += r[a]

            # If no dice left, continue letting other
            # players go until the end of the round
            left = new_state.dice_left(0)
            if left == 0:
                while not new_state.round_end():
                    for i in range(1, new_state.players):
                        new_state = baseline[i - 1](new_state, i,
                                                    baseline_model)
            else:
                for i in range(1, new_state.players):
                    new_state = baseline[i - 1](new_state, i, baseline_model)

            # Advance to the next round if needed
            if new_state.round_end():
                new_state = new_state.advance_round()

            if valid:
                # Give a reward when the game is over
                reward = new_state.cash[0] - state.cash[0]
                if new_state.game_end():
                    max_cash = max(new_state.cash)
                    if new_state.cash[0] == max_cash:
                        if new_state.cash.count(max_cash) == 1:
                            reward += constants.WIN_REWARD
                        else:
                            reward += constants.TIE_REWARD
            else:
                # Negative reward for invalid moves
                reward = -1

            # Store the transition in the learning memory
            # (remove an old observation if needed)
            memory.insert(0, (state, r, a, reward, new_state))
            if len(memory) > constants.LEARNING_MEMORY_SIZE:
                memory.pop()

            # Train on a minibatch from the learning memory
            if len(memory) >= constants.MINIBATCH_SIZE:
                mb = random.sample(memory, constants.MINIBATCH_SIZE)

                x = []
                y = []

                for (s, r, a, reward, ss) in mb:
                    x.append(state.as_vector(r))
                    if s.game_end():
                        # If the game is over, use the final reward
                        y.append(reward)
                    else:
                        # Otherwise use discounted future reward
                        p = model.predict(np.array([state.as_vector(r)]))[0]
                        p[a] = reward / constants.CASH_NORM + \
                            p[a] * constants.DISCOUNT_RATE
                        y.append(p)
                model.train_on_batch(np.array(x), np.array(y))

            state = new_state

        game_num += 1

    return model
