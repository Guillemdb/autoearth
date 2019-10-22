import numpy as np
from fragile.core.states import States
from fragile.core.models import BinarySwap


def columns_to_vector(selected, columns):
    return np.array([c in selected for c in columns]).astype(bool)


class EarthSampler(BinarySwap):

    def __init__(self, columns: list = None, force_series: list = None,
                 force_lags: list = None, p_delete: float = 0.1,
                 one_lag_per_series: bool = True, *args, **kwargs):
        super(EarthSampler, self).__init__(*args, **kwargs)
        self.columns = columns
        self.names = list(set([c.split(".")[0] for c in columns]))
        self.p_delete = p_delete
        self.soft_constraints = force_series if force_series is not None else []
        self.hard_constraints = force_lags if force_lags is not None else []
        self.name_masks = self._create_names_mask()
        self.hard_masks = self._create_hard_masks()
        self.one_lag_per_series = one_lag_per_series

    def sample(self, env_states: States = None, batch_size: int = 1,
               model_states: States = None, **kwargs) -> States:
        actions = env_states.observs.copy() if env_states is not None else np.zeros((batch_size,
                                                                                     self.dim))
        actions = self.modify_actions(actions)
        model_states.update(actions=actions)
        return model_states

    def _swap_one_action(self, actions, i):
        target_name = self.random_state.choice(self.names)

        if self.random_state.random() > self.p_delete or not actions[i].any():
            for _ in range(np.random.randint(1, 3)):
                target_mask = np.logical_not(self.name_masks[target_name])
                valid_ixs = np.arange(actions.shape[1])[self.name_masks[target_name]]
                new_val = self.random_state.choice(valid_ixs.tolist())
                if self.one_lag_per_series:
                    erased_action = np.logical_and(actions[i], target_mask)
                    erased_action[new_val] = True
                    actions[i] = np.logical_and(actions[i], erased_action)
                else:
                    erased_action = np.zeros_like(target_mask)
                    erased_action[new_val] = True
                    actions[i] = np.logical_or(actions[i], erased_action)
                target_name = self.random_state.choice(self.names)
        elif actions[i].sum() > 1:
            erased_action_mask = np.logical_not(self.name_masks[target_name])
            actions[i] = np.logical_and(actions[i], erased_action_mask)
        return actions

    def _create_hard_masks(self):
        return {const: np.array([col in self.hard_constraints
                                 for col in self.columns]).astype(bool)
                for const in self.hard_constraints}

    def _enforce_hard_constraints(self, action):

        for name in self.hard_constraints:
            mask = self.hard_masks[name]
            action = np.logical_or(action, mask)
        return action

    def _enforce_soft_constraints(self, action):
        for name in self.soft_constraints:
            mask = self.name_masks[name]
            if not action[mask].any():
                valid_ixs = np.arange(len(action))[self.name_masks[name]]
                new_val = self.random_state.choice(valid_ixs.tolist())
                action[new_val] = True
        return action

    def modify_actions(self, actions):
        actions = actions.astype(bool)
        # set action to all zeros
        for i in range(actions.shape[0]):
            actions = self._swap_one_action(actions, i)
            actions[i] = self._enforce_hard_constraints(actions[i])
            actions[i] = self._enforce_soft_constraints(actions[i])
        actions = actions.astype(int)
        return actions

    def _create_names_mask(self):
        return {n: np.array([n in t for t in self.columns]).astype(bool) for n in self.names}

    @staticmethod
    def _fix_points(points, mask):
        need_fix = np.logical_not(np.logical_and(points, mask).any(axis=1))

        def new_point_in_mask(mask):
            points = np.zeros_like(mask)
            ix = np.random.choice(np.arange(len(mask))[mask])
            points[ix] = 1
            return points
        fixes = np.array([new_point_in_mask(mask) for _ in range(need_fix.sum())])
        if len(fixes) > 0:
            points[need_fix] = np.logical_or(points[need_fix], fixes)
        return points


class EarthOneDelay(BinarySwap):

    def sample(self, env_states: States = None, batch_size: int = 1,
               model_states: States = None, **kwargs) -> States:
        actions = env_states.observs.copy() if env_states is not None else np.zeros((batch_size,
                                                                                     self.dim))
        actions = actions.astype(bool)
        # set action to all zeros
        for i in range(actions.shape[0]):
            target_name = self.random_state.choice(self.names)
            actions[i] = np.logical_and(actions[i], np.logical_not(self.namme_masks[target_name]))
            valid_ixs = np.arange(actions.shape[1])[self.name_masks[target_name]]
            actions[i, self.random_state.choice(valid_ixs)] = True
        actions = actions.astype(int)
        dt = (1 if self.dt_sampler is None else
              self.dt_sampler.calculate(batch_size=batch_size, model_states=model_states,
                                        **kwargs).astype(int))
        model_states.update(actions=actions, dt=dt)

        actions = model_states.actions.astype(bool)
        actions = np.logical_or(actions, self.force_mask)
        actions = self._enforce_soft_constraints(actions)
        model_states.update(actions=actions)
        return model_states

    def _create_names_mask(self):
        return {n: np.array([n in t for t in self.columns]).astype(bool) for n in self.names}

    def _create_force_mask(self, names):
        return np.array([n in names for n in self.names]).astype(bool)

    def _create_names_mask(self):
        return {n: np.array([n in t for t in self.names]).astype(bool) for n in self.force_name}

    def _enforce_soft_constraints(self, actions):
        acts = np.zeros_like(actions)
        for name, mask in self._name_masks.items():
            acts = np.logical_or(self._fix_points(actions, mask), acts)
        return actions

    @staticmethod
    def _fix_points(points, mask):
        need_fix = np.logical_not(np.logical_and(points, mask).any(axis=1))
        need_fix = np.logical_or(need_fix, np.logical_and(points, mask).sum() > 1)

        def new_point_in_mask(mask):
            points = np.zeros_like(mask)
            ix = np.random.choice(np.arange(len(mask))[mask])
            points[ix] = 1
            return points
        fixes = np.array([new_point_in_mask(mask) for _ in range(need_fix.sum())])
        if len(fixes) > 0:
            points[need_fix] = np.logical_and(points[need_fix], fixes)
        return points




