from humanoid_climb.stances.base_stance import Stance

_STANCES_ = []

STANCE_NONE = Stance(stance=[-1, -1, -1, -1],
                     previous_stance=None,
                     state_file=None,
                     action_override=[-1, -1, -1, -1],
                     exclude_targets=[])

STANCE_1 = Stance(stance=[10, 9, -1, -1],
                  previous_stance=None,
                  state_file=None,
                  action_override=[-1, -1, -1, -1],
                  exclude_targets=[])

STANCE_2 = Stance(stance=[10, 9, 2, -1],
                  previous_stance=[10, 9, -1, -1],
                  state_file="/states/state_10_9_n_n.npz",
                  action_override=[1, 1, -1, -1],
                  exclude_targets=[])

STANCE_3 = Stance(stance=[10, 9, 2, 1],
                  previous_stance=[10, 9, 2, -1],
                  state_file="/states/state_10_9_2_n.npz",
                  action_override=[1, 1, 1, -1],
                  exclude_targets=[])

STANCE_4 = Stance(stance=[10, 13, 2, 1],
                  previous_stance=[10, 9, 2, 1],
                  state_file="/states/state_10_9_2_1.npz",
                  action_override=[1, -1, 1, 1],
                  exclude_targets=[9])

STANCE_5 = Stance(stance=[10, 13, 2, 5],
                  previous_stance=[10, 13, 2, 1],
                  state_file="/states/state_10_13_2_1.npz",
                  action_override=[1, 1, 1, -1],
                  exclude_targets=[1])

STANCE_6 = Stance(stance=[13, 13, -1, 5],
                  previous_stance=[10, 13, 2, 5],
                  state_file="/states/state_10_13_2_5.npz",
                  action_override=[-1, 1, -1, 1],
                  exclude_targets=[10])

_STANCES_.append(STANCE_NONE)
_STANCES_.append(STANCE_1)
_STANCES_.append(STANCE_2)
_STANCES_.append(STANCE_3)
_STANCES_.append(STANCE_4)
_STANCES_.append(STANCE_5)
_STANCES_.append(STANCE_6)


def set_root_path(root_path):
    for stance in _STANCES_:
        stance.root_path = root_path
