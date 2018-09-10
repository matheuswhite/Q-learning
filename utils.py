

class QfunctionRow:

    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward

    def __eq__(self, other):
        return self.state == other.state and self.action == other.action

    def __gt__(self, other):
        return self.reward > other.reward

    def __lt__(self, other):
        return self.reward < other.reward


class Qfunction:

    def __init__(self):
        self.table = []

    def find_state_action(self, value):
        for x in range(0, len(self.table)):
            if value == self.table[x]:
                return x
        return None

    def update_expected_return(self, state, action, reward):
        row = QfunctionRow(state, action, reward)
        index = self.find_state_action(row)
        if index is not None:
            self.table[index].reward = reward
        else:
            self.table.append(row)

    def get_expected_return(self, state, action):
        row = QfunctionRow(state, action, 0)
        index = self.find_state_action(row)
        if index is not None:
            return self.table[index].reward
        else:
            return 0