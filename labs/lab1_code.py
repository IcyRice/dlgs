# imports
import random
from typing import List, Tuple
import time
from copy import deepcopy  # world -> thought


def run():
    agents = (Agent('O'), Human('X'))
    game = Game(agents)
    game.play()


# world and world model
class State:
    def __init__(self, cols=7, rows=6, win_req=4):
        self.board = [['.'] * cols for _ in range(rows)]
        self.heights = [1] * cols
        self.num_moves = 0
        self.win_req = win_req

    def get_avail_actions(self) -> List[int]:
        return [i for i in range(len(self.board[0])) if self.heights[i] <= len(self.board)]

    def put_action(self, action, agent):
        self.board[len(self.board) - self.heights[action]][action] = agent.name
        self.heights[action] += 1
        self.num_moves += 1

    def is_over(self):
        return self.num_moves >= len(self.board) * len(self.board[0])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        header = " ".join([str(i) for i in range(len(self.board[0]))])
        line = "".join(["-" for _ in range(len(header))])
        board = [[e for e in row] for row in self.board]
        board = '\n'.join([' '.join(row) for row in board])
        return '\n' + header + '\n' + line + '\n' + board + '\n'


# evaluate the utility of a state
def utility(state: 'State'):
    board = state.board
    n_cols = len(board[0]) - 1
    n_rows = len(board) - 1

    def diags_pos():
        """Get positive diagonals, going from bottom-left to top-right."""
        for di in ([(j, i - j) for j in range(n_cols)] for i in range(n_cols + n_rows - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < n_cols and j < n_rows]

    def diags_neg():
        """Get negative diagonals, going from top-left to bottom-right."""
        for di in ([(j, i - n_cols + j + 1) for j in range(n_cols)] for i in range(n_cols + n_rows - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < n_cols and j < n_rows]

    cols = list(map(list, list(zip(*board))))
    rows = board
    diags = list(diags_neg()) + list(diags_pos())
    lines = rows + cols + diags
    strings = ["".join(s) for s in lines]
    for string in strings:
        if 'OOOO' in string:
            return -1
        if 'XXXX' in string:
            return 1
    return 0


# parrent class for mcts, minmax, human, and any other idea for an agent you have
class Agent:
    def __init__(self, name: str):
        self.name: str = name

    def get_action(self, state: State):
        return random.choice(state.get_avail_actions())


class Human(Agent):
    def __init__(self, name):
        super(Human, self).__init__(name)

    def get_action(self, state: State):
        # return super().get_action(state)
        a = state.get_avail_actions()
        #userMove = int(input("enter move 0-6: "))
        userMove = input("enter move 0-6: ")
        if int(userMove) in a:
            return int(userMove)
        else:
            print("Invalid Move")
            self.get_action(state)


# class Gekko(Agent):
#    def __init__(self, name):
#        super(Gekko, self).__init__(name)


# class MinMax(Agent):
#   def __init__(self, name):
#       super(MinMax, self).__init__(name)


# class Node:
#     def __init__(self, state: State, parent: 'Node' = None):
#         self.children: List['Nodes'] = []
#         self.parent: 'Node' = parent
#         self.state: State = state


# class MCTS(Agent):
#    def __init__(self, name):
#        super(MCTS, self).__init__(name)


# connecting states and agents
class Game:
    def __init__(self, agents: Tuple[Agent]):
        self.agents = agents
        self.state = State()

    def play(self):
        while utility(self.state) == 0 and not self.state.is_over():
            for agent in self.agents:
                if utility(self.state) == 0 and not self.state.is_over():
                    action = agent.get_action(self.state)
                    self.state.put_action(action, agent)
                    print(self.state)
        print("GAME OVER")


#agents = (Agent('O'), Agent('X'))
#game = Game(agents)
# game.play()

run()
