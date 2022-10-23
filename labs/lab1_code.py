# imports
import random
import math
from typing import List, Tuple
import time
from copy import deepcopy  # world -> thought


def run():
    agents = (Agent('O'), Human('X'))
    game = Game(agents)
    game.play()


def testUtil():
    agents = (Agent('O'), Gekko('X'))
    game = Game(agents)
    game.play()


def testGekko():
    agents = (Gekko('O'), Gekko('X'))
    game = Game(agents)
    game.play()

def testGekkoMinmax():
    agents = (Gekko('O'), MinMax('X'))
    game = Game(agents)
    game.play()

def testMinmaxHuman():
    agents = (MinMax('O'), Human('X'))
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


# evaluate the utility of a state - this is really just a terminal check
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
        a = state.get_avail_actions()
        userMove = input("enter move 0-6: ")
        if int(userMove) in a:
            return int(userMove)
        else:
            print("Invalid Move")
            self.get_action(state)


class Gekko(Agent):
    def __init__(self, name):
        super(Gekko, self).__init__(name)
        if self.name == 'X':
            self.targetUtil = 1
        else:
            self.targetUtil = -1


    def get_action(self, state: State):
        legal_actions = state.get_avail_actions()
        action = -1

        for a in legal_actions:
            newState = deepcopy(state)
            newState.put_action(a, self)
            newUtil = self.gekko_utility(newState)
            if newUtil == self.targetUtil:          # a move that wins
                action = a
                print(self.name, "-Gekko plays move for 4 in a row - and now wins! util = ", newUtil)
                break
            elif newUtil == self.targetUtil * 2:    # a move that gives the agent 3 in a row
                action = a
                print(self.name, "-Gekko plays move for 3 in a row, util = ", newUtil)

        if action == -1:
            action = random.choice(legal_actions)
            print(self.name, "-Gekko plays a random move: ", action)
        return action


    def gekko_utility(self, state: State):
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
            if 'OOO' in string:
                return -2
            if 'XXX' in string:
                return 2
        return 0


    def get_action2(self, state: State):  # Gekko is currently Human for debugging
        # utility(state)
        a = state.get_avail_actions()
        print(self.gekko_utility2(state))
        print(a)

        userMove = input("enter move 0-6: ")
        if int(userMove) in a:
            return int(userMove)
        else:
            print("Invalid Move")
            self.get_action2(state)


    def get_action3(self, state: State):
        actions = state.get_avail_actions()
        bestUtil = 0
        selectedAction = -1
        print(state)

        for a in actions:
            newState = deepcopy(state)
            # print(self.gekko_utility(newState))
            newState.put_action(a, self)
            #print(self.gekko_utility(newState))
            print(newState)
            newStateUtil = self.gekko_utility(newState)
            print(newStateUtil)
            print("__________________")

            if newStateUtil < bestUtil:     # Gekko plays 'O'
                bestUtil = newStateUtil
                selectedAction = a

        if selectedAction > -1:
            print("Gekko found winning action!")
            return selectedAction
        else:
            print("Gekko selecting random action")
            return random.choice(actions)


    def gekko_utility2(self, state: State):
        #tempState = deepcopy(state)
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

        return lines


    def select_move(self, state):
        # keep the utility lists structured so we can retrace the column index
        # make a greedy move to look for 3 connected with open neighbour
        # if non exists, look for 2 connected with open neighbour
        # return column index (avail move) for the open neighbour position
        # random choice for catch-all

        # board = state.board             # 2d list
        # max X value (columns also represent action space)
        #n_cols = len(board[0]) - 1
        # n_rows = len(board) - 1         # max Y

        lines = self.gekko_utility(state)
        strings = ["".join(s) for s in lines]
        for string in strings:
            if 'OOO.' in string:
                return -1
            if 'XXX.' in string:
                return 1

        return  # work in progress



class MinMax(Agent):
    def __init__(self, name):
        super(MinMax, self).__init__(name)
        if self.name == 'X':
            self.isMax = True
        else:
            self.isMax = False
        self.depth_limit = 2    # careful with this

    def get_action(self, state: State):
        self.recursionCounter = 0
        actions = state.get_avail_actions()
        currentVal = 0
        currentAction = -1

        if self.isMax:
            for a in actions:
                newState = deepcopy(state)
                newState.put_action(a, self)
                val = self.minimax(newState, 0, self.isMax)
                if val > currentVal:
                    currentVal = val
                    currentAction = a
        else:
            for a in actions:
                newState = deepcopy(state)
                newState.put_action(a, self)
                val = self.minimax(newState, 0, self.isMax)
                if val < currentVal:
                    currentVal = val
                    currentAction = a
        if currentAction == -1:
            currentAction = random.choice(actions)
            print(self.name, "-MinMax plays a random action: ", currentAction)
        print("Minmax plays: ", currentAction, " | recursions: ", self.recursionCounter)
        return currentAction


    def maxUtil():
        return

    def minUtil():
        return

    def minimax(self, state: State, depth, isMax):
        self.recursionCounter += 1
        value = utility(state)        
        if value == 1 or value == -1: # reached terminal state
            #print("Hit terminal state with util: ", value, " isMax? ", isMax)
            return value

        if depth >= self.depth_limit:
            #print("Hit search depth limit: ", depth, " value = ", value)
            return value

        # MAX
        if isMax:
            actions = state.get_avail_actions()
            for a in actions:
                newState = deepcopy(state)
                newState.put_action(a, self)
                actionValue = self.minimax(newState, depth + 1, not isMax)

                if actionValue > value: # maximizing
                    value = actionValue
        # MIN
        else:
            actions = state.get_avail_actions()
            for a in actions:
                newState = deepcopy(state)
                newState.put_action(a, self)
                actionValue = self.minimax(newState, depth + 1, not isMax)

                if actionValue < value: # minimizing
                    value = actionValue

        return value



class Node:
    def __init__(self, state: State, parent: 'Node' = None):
        self.children: List['Nodes'] = []
        self.parent: 'Node' = parent
        self.state: State = state


class MCTS(Agent):
    def __init__(self, name):
        super(MCTS, self).__init__(name)


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


# run()
# testUtil()
#testGekko()
testGekkoMinmax()
#testMinmaxHuman()