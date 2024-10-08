import gymnasium as gym
import numpy as np
import pygame


class XsAndOs(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = 3  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.size, self.size), dtype=np.int32)

        # We have 9 actions
        self.action_space = gym.spaces.Discrete(9)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._grid

    def _get_info(self):
        return {}

    def reset(self, seed=None, options={"start_turn": "o"}):
        super().reset(seed=seed)

        self._grid = np.zeros((self.size, self.size), dtype=np.int32)
        self._turn = 0 if options["start_turn"] == "o" else 1
        self._won = False
        self._end = False
        self._wrong_action_has_been_played = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _check_end(self):
        end = len(self._grid[self._grid == 0]) == 0
        return end

    def _check_win_condition(self):
        vertical = np.sum(self._grid, axis=0)
        horizontal = np.sum(self._grid, axis=1)
        diag = np.array([np.sum(self._grid.diagonal())])
        anti_diag = np.array([np.sum(np.fliplr(self._grid).diagonal())])

        values = np.concatenate((vertical, horizontal, diag, anti_diag))

        return 3 in values or -3 in values

    def step(self, action):
        if self._wrong_action_has_been_played:
            return self._get_obs(), -1, True, False, {"status": "player_" + str(self._turn) + " played a wrong action."}

        if self._won:
            return self._get_obs(), 1, True, False, {}

        if self._end:
            return self._get_obs(), 0, True, False, {}

        if self._grid[action // 3, action % 3] != 0:
            self._wrong_action_has_been_played = True
            return self._get_obs(), 0, True, False, {"status": "player_" + str(self._turn) + " played a wrong action."}

        self._grid[action // 3, action % 3] = -1 if self._turn else 1
        self._turn = (self._turn + 1) % 2

        win_game = self._check_win_condition()
        end_game = self._check_end()
        terminated = win_game or end_game

        reward = 0

        if win_game:
            self._won = win_game
            reward = -0.5
        elif end_game:
            self._end = end_game
            reward = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((250, 250, 250))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Now we draw the agent
        for i in range(self.size):
            for j in range(self.size):
                pos = (np.array([i, j]) + 0.5) * pix_square_size

                if self._grid.T[i, j] == 1:
                    pygame.draw.circle(
                        canvas, (0, 0, 255),
                        pos,
                        pix_square_size / 3,
                        width = 10
                    )
                if self._grid.T[i, j] == -1:
                    pygame.draw.lines(canvas, (255, 0, 0), True, [(pos[0]-50,pos[1]-50),(pos[0]+50,pos[1]+50)], 10)
                    pygame.draw.lines(canvas, (255, 0, 0), True, [(pos[0]-50,pos[1]+50),(pos[0]+50,pos[1]-50)], 10)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas, 0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas, 0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:
            # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
