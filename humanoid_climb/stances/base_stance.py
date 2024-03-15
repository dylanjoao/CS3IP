class Stance():
    def __init__(self, previous_stance, state_file, stance, action_override, exclude_targets):
        self.root_path = None
        self.previous_stance = previous_stance
        self.state_file = state_file
        self.stance = stance
        self.action_override = action_override
        self.exclude_targets = exclude_targets

    def get_args(self):
        state_path = self.root_path+self.state_file if self.state_file is not None else None
        dict = {'motion_path': self.stance,
                'state_file': state_path,
                'action_override': self.action_override,
                'motion_exclude_targets': self.exclude_targets}
        return dict
