from __future__ import division

import collections
import copy
import random

from keras.layers import Dense, Activation
from keras.models import Sequential

from . import constants

SpotTuple = collections.namedtuple('Spot', [
    'dice',
    'bills'
])
StateTuple = collections.namedtuple('State', [
    'players',
    'round_num',
    'cash',
    'bill_deck',
    'spots'
])

class Spot(SpotTuple):
    pass

class State(StateTuple):
    def deal_bills(self):
        i = 0
        bills = [[], [], [], [], [], []]
        for i in range(6):
            while sum(bills[i]) < 50:
                bills[i].append(self.bill_deck.pop())
            while len(bills[i]) < 5:
                bills[i].append(0)
            bills[i].sort(reverse=True)

        return bills

    def promote_player(self, player):
        """
        Move a player at a given position to be the first player
        This is because our network assumes it is player 0
        This transformation does not affect the semantics of the game
        """
        state = copy.deepcopy(self)
        state = state._replace(cash=[state.cash.pop(player)] + state.cash)

        new_spots = []
        for spot in state.spots:
            new_spots.append(spot._replace(dice=[spot.dice.pop(player)] + spot.dice))
        state = state._replace(spots=new_spots)

        return state

    def dice_left(self, player):
        """
        Return the number of dice still available to be rolled by a player
        """
        dice = [spot.dice for spot in self.spots]
        total_used = [sum(dice[j][i] for j in range(6))
                for i in range(self.players)]
        return constants.MAX_DICE - total_used[player]

    def round_end(self):
        """
        Check if a round has ended (all players have used all their dice)
        """
        return all(self.dice_left(i) == 0 for i in range(self.players))

    def advance_round(self):
        """
        Create a new state for the beginning of the next round
        """
        # Make sure the round is over
        assert self.round_end()

        # Copy the previous state and increment the round
        state = copy.deepcopy(self)
        if state.round_num < constants.MAX_ROUNDS:
            state = state._replace(round_num=state.round_num + 1)

        # Award new cash to all players
        new_cash = state.cash
        for spot in state.spots:
            # Record the number of times we see each count
            # of dice so that we are able to check for ties
            counts = collections.defaultdict(lambda: 0)
            for count in spot.dice:
                counts[count] += 1

            # Award the bills in the correct order
            ranked_dice = sorted(enumerate(spot.dice), key=lambda x: x[1])
            while len(ranked_dice) > 0:
                player, count = ranked_dice.pop()

                # This player only gets a bill if their count is unique
                if counts[count] == 1:
                    try:
                        new_cash[player] += spot.bills.pop(0)
                    except IndexError:
                        break
        state = state._replace(cash=new_cash)

        # If this was not the last round, deal a new set of bills
        if state.round_num != constants.MAX_ROUNDS:
            bills = self.deal_bills()
            for (i, spot) in enumerate(state.spots):
                state.spots[i] = spot._replace(dice=[0] * state.players, bills=bills[i])

        return state

    def game_end(self):
        """
        Check if the game has ended (last round has finished)
        """
        return self.round_num == constants.MAX_ROUNDS and self.round_end()

    def as_vector(self, roll):
        """
        Return the state as a floating point vector
        for feeding as input into the neural network
        """
        vec = []

        # Number of players
        vec.append(self.players / constants.MAX_PLAYERS)

        # Round number
        vec.append(self.round_num / constants.MAX_ROUNDS)

        # Cash for each players (normalize by large value)
        vec.extend(c / constants.CASH_NORM for c in self.cash)

        for i in range(6):
            # Dice on each spot
            vec.extend(d / constants.MAX_DICE for d in self.spots[i].dice)

            # Bills on each spot
            vec.extend(b / max(constants.INIT_BILLS) for b in self.spots[i].bills)

        vec.extend(n / constants.MAX_DICE for n in roll)

        return vec

    @staticmethod
    def initial(players=constants.MAX_PLAYERS):
        """
        Initialize state for a new game
        """
        # Create a new shuffled deck
        bill_deck = copy.copy(constants.INIT_BILLS)
        random.shuffle(bill_deck)

        state = State(players=players,
                      round_num=1,
                      cash=[0] * players,
                      bill_deck=bill_deck,
                      spots=None)

        # Deal out the initial bills
        bills = state.deal_bills()
        state = state._replace(spots=[
            Spot(dice=[0] * players, bills=bills[0]),
            Spot(dice=[0] * players, bills=bills[1]),
            Spot(dice=[0] * players, bills=bills[2]),
            Spot(dice=[0] * players, bills=bills[3]),
            Spot(dice=[0] * players, bills=bills[4]),
            Spot(dice=[0] * players, bills=bills[5])
        ])
        return state

def roll(count):
    dice = [random.randint(1, 6) for _ in range(count)]
    return [dice.count(i + 1) for i in range(6)]

def get_dimensions(players):
    # Assume players is fixed
    players = 5

    dimensions = 0

    # Player num
    dimensions += 1

    # Round num
    dimensions += 1

    # Cash vector
    dimensions += players

    for _ in range(6):
        # Dice count for each player
        dimensions += players

        # Bills on each spot (max 5)
        dimensions += 5

    # Number of dice of each value in the roll
    dimensions += 6

    return dimensions

def create_model(players=constants.MAX_PLAYERS,
                 kernel_initializer='lecun_uniform',
                 activation='sigmoid',
                 optimizer='adam'):
    """
    Construct a new neural net ready for training
    """
    dims = get_dimensions(players)
    model = Sequential([
        Dense(dims, kernel_initializer=kernel_initializer, input_shape=(dims,)),
        Activation(activation),
        Dense(dims // 2, kernel_initializer=kernel_initializer),
        Activation(activation),
        Dense(6, kernel_initializer=kernel_initializer),
        Activation('linear')
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
