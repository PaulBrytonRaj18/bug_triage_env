# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Bug Triage Env — OpenEnv-compatible Environment Interface.

This module provides the OpenEnv Environment wrapper for the Bug Triage
environment, bridging the FastAPI HTTP server with the openenv-core interface.
"""

from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    Environment = object
    State = None

try:
    from ..models import BugAction, BugObservation
except ImportError:
    from models import BugAction, BugObservation

from .environment import BugTriageEnvironment


class BugTriageEnvEnvironment(Environment):
    """
    OpenEnv-compatible wrapper for the Bug Triage Environment.

    This class wraps BugTriageEnvironment to expose the openenv-core
    Environment interface, enabling integration with openenv validate
    and openenv CLI tools.

    Example:
        >>> env = BugTriageEnvEnvironment()
        >>> obs = env.reset()
        >>> print(obs.issue_id)  # "ISS-1000"
        >>>
        >>> action = BugAction(
        ...     action_type="label_bug",
        ...     severity="P1",
        ...     issue_id=obs.issue_id,
        ... )
        >>> obs = env.step(action)
        >>> print(obs.cumulative_score)  # 0.9
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        """Initialize the OpenEnv wrapper."""
        self._env = BugTriageEnvironment()
        self._state = State(episode_id=str(uuid4()), step_count=0) if State else None

    def reset(self, task_id: str = "easy") -> BugObservation:
        """
        Reset the environment for the given task.

        Args:
            task_id: One of 'easy', 'medium', 'hard'

        Returns:
            BugObservation with the first issue
        """
        obs = self._env.reset(task_id=task_id)
        if self._state:
            self._state = State(
                episode_id=str(uuid4()),
                step_count=0,
            )
        return obs

    def step(self, action: BugAction) -> BugObservation:  # type: ignore[override]
        """
        Execute one triage step.

        Args:
            action: BugAction with the triage decision

        Returns:
            BugObservation with the next issue and reward info
        """
        obs = self._env.step(action)
        if self._state:
            self._state = State(
                episode_id=self._state.episode_id,
                step_count=self._env.state.step_count,
            )
        return obs

    @property
    def state(self):
        """Return the current episode state."""
        if self._state:
            return self._state
        return self._env.state

    @property
    def triaged(self):
        """Return the list of triaged actions this episode."""
        return self._env.triaged
