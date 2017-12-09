import collections
import copy
import random

import numpy as np

from .state import roll

def winning_cash(state, player, spot_num, dice):
    """
    Produce the amount of cash a player would currently
    win if they placed their dice on the given spot
    """
    spot = state.spots[spot_num]
    new_dice = copy.copy(spot.dice)
    spot_bills = copy.copy(spot.bills)
    new_dice[player] += dice

    # Accumulate the different dice counts on each spot
    counts = collections.defaultdict(lambda: 0)
    for count in new_dice:
        counts[count] += 1

    # Count the payout for this player
    ranked_dice = sorted(enumerate(spot.dice), key=lambda x: x[1])
    while len(ranked_dice) > 0:
        p, count = ranked_dice.pop()
        if p == player:
            return spot_bills.pop(0)

    return 0

def greedy_turn(state, player, _model=None):
    """
    Pick a number greedily based on the cash the player would currently win
    """
    new_state = copy.deepcopy(state)
    left = new_state.dice_left(player)
    if left == 0:
        return new_state

    player_roll = roll(left)
    max_gain = -1
    max_spot = 0
    for (i, r) in enumerate(player_roll):
        if r > 0:
            gain = winning_cash(new_state, player, i, r)
            if gain > max_gain:
                max_gain = gain
                max_spot = i

    new_state.spots[max_spot].dice[player] += player_roll[max_spot]
    return new_state

def random_turn(state, player, _model=None):
    """
    Roll the dice for the given player and choose randomly
    """
    new_state = copy.deepcopy(state)

    # If the player has no dice to roll, the state does not change
    left = state.dice_left(player)
    if left == 0:
        return new_state

    # Roll the dice and pick any non-zero value
    player_roll = roll(left)
    i = 0
    while player_roll[i] == 0:
        i = random.randint(0, 5)

    # Update the dice at the given spot and return
    new_state.spots[i].dice[player] += player_roll[i]
    return new_state

def choose_action(predicted, roll):
    """
    Choose the first valid action from a vector of scores
    """
    in_order = sorted(enumerate(predicted), key=lambda x: x[1])
    a = in_order.pop()[0]
    while roll[a] == 0:
        a = in_order.pop()[0]

    return a

def model_turn(state, player, model):
    """
    Choose an action based on a trained model
    """
    left = state.dice_left(player)
    if left == 0:
        return copy.deepcopy(state)

    # Force our player to position 0
    if player != 0:
        state = state.promote_player(player)

    # Pick an action
    r = roll(left)
    p = model.predict(np.array([state.as_vector(r)]))[0]
    a = choose_action(p, r)
    state = copy.deepcopy(state)
    state.spots[a].dice[0] += r[a]

    # Put the players back in order
    if player != 0:
        state = state.promote_player(1)

    return state

def biggest_turn(state, player, _model=None):
    """
    Always pick the die with the largest count
    """
    new_state = copy.deepcopy(state)
    left = state.dice_left(player)
    if left == 0:
        return new_state

    # Pick the die with the largest count
    r = roll(left)
    a = max(enumerate(r), key=lambda x: x[1])[0]
    new_state.spots[a].dice[player] += r[a]

    return new_state

def richest_turn(state, player, _model=None):
    """
    Always pick the spot with the highest bill
    """
    new_state = copy.deepcopy(state)
    left = state.dice_left(player)
    if left == 0:
        return new_state

    # Pick the spot with the largest bill
    r = roll(left)
    valid_pos = [pos for pos in enumerate(r) if pos[1] > 0]
    a = max(valid_pos, key=lambda x: state.spots[x[0]].bills[0])[0]
    new_state.spots[a].dice[player] += r[a]

    return new_state
