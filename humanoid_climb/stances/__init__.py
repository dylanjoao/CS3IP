from humanoid_climb.stances.base_stance import Stance

_STANCES_ = []

STANCE_NONE = Stance(None,
                     None,
                     [[10, 9, -1, -1]],
                     [-1, -1, -1, -1],
                     [[]])
_STANCES_.append(STANCE_NONE)

STANCE_4 = Stance([10,9,2,1],
                  "/states/state_10_9_2_1_v3.npz",
                  [[10,13,2,1]],
                  [1, 1, 1, 1],
                  [[9]])
_STANCES_.append(STANCE_4)

def set_root_path(root_path):
    for stance in _STANCES_:
        stance.root_path = root_path