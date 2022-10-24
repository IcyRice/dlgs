# imports
import random
import math
from typing import ForwardRef, List, Tuple
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


def testMinmaxGekko():
    agents = (MinMax('O'), Gekko('X'))
    game = Game(agents)
    game.play()


def testMinmaxHuman():
    agents = (MinMax('O'), Human('X'))
    game = Game(agents)
    game.play()


def testGekkoMCTS():
    agents = (Gekko('O'), MCTS('X'))
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
                print(self.name, "-Gekko plays move for 4 in a row: ",
                      a, " - and now wins! | util = ", newUtil)
                break
            elif newUtil == self.targetUtil * 2:    # a move that gives the agent 3 in a row
                action = a
                print(self.name, "-Gekko plays move for 3 in a row: ",
                      a, " | util = ", newUtil)

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


class MinMax(Agent):
    def __init__(self, name):
        super(MinMax, self).__init__(name)
        if self.name == 'X':
            self.isMax = True
        else:
            self.isMax = False
        self.depth_limit = 2    # careful with this, the search-tree goes bonkers

    def get_action(self, state: State):
        self.recursionCounter = 0
        actions = state.get_avail_actions()
        currentVal = 0
        currentAction = -1  # default invalid action

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
        print(self.name, "-Minmax plays: ", currentAction,
              " | recursions: ", self.recursionCounter)
        return currentAction

    def minimax(self, state: State, depth, isMax):
        self.recursionCounter += 1
        value = utility(state)
        if value == 1 or value == -1:  # reached terminal state
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

                if actionValue > value:  # maximizing
                    value = actionValue
        # MIN
        else:
            actions = state.get_avail_actions()
            for a in actions:
                newState = deepcopy(state)
                newState.put_action(a, self)
                actionValue = self.minimax(newState, depth + 1, not isMax)

                if actionValue < value:  # minimizing
                    value = actionValue

        return value


class Node:
    def __init__(self, state: State, parent: 'Node' = None):
        self.children: List['Nodes'] = []
        self.parent: 'Node' = parent
        self.state: State = state
        self.action = -1
        self.visitCount = 0
        self.value = 0
        self.depth = 0
        self.isLeaf = False


class MCTS(Agent):
    def __init__(self, name):
        super(MCTS, self).__init__(name)
        self.epsilon = 0.1  # exploitation vs exploration
        self.maxSteps = 10

    def get_action(self, state: State):
        return self.search(state)

    def search(self, state: State):
        root = Node(state)
        steps = 0
        while steps < self.maxSteps:
            steps += 1
            v1 = self.select(root)
            reward = self.simulate(steps, v1)
            self.backprop(v1, reward)
        # return max(root.children)
        move = self.bestChild(root)
        print(self.name, "-MCTS plays: ", move)
        return move

    # creates and selects a new child if the parent node has not explored its actions yet
    # returns random child if parent is already explored
    def select(self, parent: Node):
        actions = parent.state.get_avail_actions()
        for a in actions:
            if len(parent.children) < a:
                return self.expand(parent, a)
        print("select(): node has all children, return random")
        return random.choice(parent.children)

    # takes a node in the tree with incomplete list of children and initializes the next child

    def expand(self, node: Node, a):
        # assumes an incomplete list of children
        # the [len(node.children)]-index should reflect the action we need to expand
        #action = node.state.get_avail_actions()[len(node.children)]
        action = a
        newState = deepcopy(node.state)
        newState.put_action(action, self)       # s1 = (s0, a)
        # initialize the new action/child-node with its state
        child = Node(newState, node)
        child.action = action
        node.children.append(child)
        return child

    def simulate(self, depth, node: Node):
        s = node.state
        while utility(s) == 0 and depth < self.maxSteps:
            actions = s.get_avail_actions()
            a = random.choice(actions)
            s = self.forward(s, a)

        terminalUtil = utility(s)
        print("simulate(): stopping sim with util: ",
              terminalUtil, " | at depth: ", depth)
        return terminalUtil

    def forward(self, state: State, action):
        newState = deepcopy(state)
        newState.put_action(action, self)
        return newState

    def bestChild(self, node: Node):  # returns the action
        values = []
        bestVal = -1
        bestAction = -1
        for c in range(len(node.children)):
            values.append(node.children[c].value)
            if node.children[c].value > bestVal:
                bestVal = node.children[c].value
                bestAction = node.children[c].action
        print(values)
        return bestAction

    def backprop(self, node: Node, value):
        node.visitCount += 1
        node.value += value
        if node.parent is not None:
            self.backprop(node.parent, value)


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


# run()
# testUtil()
# testGekko()
# testGekkoMinmax()
# testMinmaxGekko()
# testMinmaxHuman()
testGekkoMCTS()
