from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import BGSConfig
from .distance import DistanceEstimator


@dataclass
class BagState:
    bag_id: int
    owner_id: Optional[int] = None
    owner_since: float = 0.0
    last_close_time: float = 0.0
    candidate_owner: Optional[int] = None
    candidate_since: float = 0.0
    owner_distance_history: Deque[float] = None
    status: str = "OK"

    def __post_init__(self):
        if self.owner_distance_history is None:
            self.owner_distance_history = deque(maxlen=BGSConfig.DISTANCE_HISTORY_SIZE)


class OwnershipManager:
    def __init__(self, distance_estimator: DistanceEstimator):
        self.distance_estimator = distance_estimator
        self.config = BGSConfig
        self.bag_states: Dict[int, BagState] = {}

    def update_ownership(self, bags: List[Dict], people: List[Dict], current_time: float) -> Dict[int, BagState]:
        active_bag_ids = {bag['id'] for bag in bags}
        self.bag_states = {k: v for k, v in self.bag_states.items() if k in active_bag_ids}

        for bag in bags:
            if bag['id'] not in self.bag_states:
                self.bag_states[bag['id']] = BagState(bag_id=bag['id'], last_close_time=current_time)

        for bag in bags:
            bag_id = bag['id']
            bag_state = self.bag_states[bag_id]
            bag_pos = bag['position_3d']

            person_distances = []
            for person in people:
                person_id = person['id']
                person_pos = person['position_3d']
                distance = self.distance_estimator.calculate_distance(bag_pos, person_pos)
                person_distances.append((person_id, distance))

            person_distances.sort(key=lambda x: x[1])

            owner_distance = None
            if bag_state.owner_id is not None:
                for pid, dist in person_distances:
                    if pid == bag_state.owner_id:
                        owner_distance = dist
                        break

            sample_distance = owner_distance if owner_distance is not None else (person_distances[0][1] if person_distances else 999.0)
            bag_state.owner_distance_history.append(sample_distance)
            closest_distance = person_distances[0][1] if person_distances else 999.0

            self._apply_ownership_rules(
                bag_state,
                person_distances,
                closest_distance,
                owner_distance,
                current_time,
            )
            self._update_bag_status(bag_state, current_time)

        return self.bag_states

    def _apply_ownership_rules(
        self,
        bag_state: BagState,
        person_distances: List[Tuple[int, float]],
        closest_distance: float,
        owner_distance: Optional[float],
        current_time: float,
    ):
        if not person_distances:
            return

        closest_person_id, _ = person_distances[0]

        if bag_state.owner_id is None:
            if closest_distance <= self.config.ASSIGNMENT_DISTANCE:
                if bag_state.candidate_owner == closest_person_id:
                    time_as_candidate = current_time - bag_state.candidate_since
                    if time_as_candidate >= self.config.CONFIRMATION_TIME:
                        bag_state.owner_id = closest_person_id
                        bag_state.owner_since = current_time
                        bag_state.last_close_time = current_time
                        bag_state.candidate_owner = None
                else:
                    bag_state.candidate_owner = closest_person_id
                    bag_state.candidate_since = current_time
            else:
                bag_state.candidate_owner = None

        release_after = self.config.UNATTENDED_THRESHOLD + self.config.OWNERSHIP_RELEASE_GRACE
        if (current_time - bag_state.last_close_time >= release_after) and owner_distance is None:
            bag_state.owner_id = None
            bag_state.owner_since = 0.0
            bag_state.candidate_owner = None
            bag_state.owner_distance_history.clear()
            return

        if owner_distance is not None and owner_distance <= self.config.ASSIGNMENT_DISTANCE:
            bag_state.last_close_time = current_time

        time_locked = current_time - bag_state.owner_since
        if time_locked < self.config.OWNERSHIP_LOCK_TIME:
            return

        owner_close = owner_distance is not None and owner_distance <= self.config.ASSIGNMENT_DISTANCE
        if owner_close or closest_distance > self.config.ASSIGNMENT_DISTANCE:
            return

        if owner_distance is None:
            bag_state.owner_id = closest_person_id
            bag_state.owner_since = current_time
            bag_state.last_close_time = current_time
            bag_state.owner_distance_history.clear()
            return

        if owner_distance - closest_distance >= self.config.SWITCH_DISTANCE_IMPROVEMENT:
            bag_state.owner_id = closest_person_id
            bag_state.owner_since = current_time
            bag_state.last_close_time = current_time
            bag_state.owner_distance_history.clear()

    def _update_bag_status(self, bag_state: BagState, current_time: float):
        if bag_state.owner_distance_history:
            smoothed_owner_distance = float(np.mean(list(bag_state.owner_distance_history)))
            if smoothed_owner_distance <= self.config.ASSIGNMENT_DISTANCE:
                bag_state.last_close_time = current_time

        time_since_close = current_time - bag_state.last_close_time
        if time_since_close >= self.config.UNATTENDED_THRESHOLD:
            bag_state.status = "UNATTENDED"
        elif time_since_close >= self.config.POTENTIAL_THRESHOLD:
            bag_state.status = "POTENTIAL"
        else:
            bag_state.status = "OK"