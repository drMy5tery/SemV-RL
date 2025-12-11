import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# TD(0) Algorithm Implementation with Visualizations\n",
    "\n",
    "This notebook implements the TD(0) algorithm for Temporal Difference learning in a GridWorld environment. It includes visualizations for the environment, value function heatmaps, and learning curves."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import matplotlib.animation as animation\n",
    "from IPython.display import HTML\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. GridWorld Environment\n",
    "\n",
    "We define a simple 5x5 GridWorld environment. \n",
    "- **Start State**: (0, 0)\n",
    "- **Goal State**: (4, 4)\n",
    "- **Obstacles**: Defined manually\n",
    "- **Actions**: Up, Down, Left, Right\n",
    "- **Rewards**: +10 at Goal, -1 per step otherwise (to encourage shortest path)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class GridWorld:\n",
    "    def __init__(self, size=5):\n",
    "        self.size = size\n",
    "        self.state = (0, 0)\n",
    "        self.goal = (size-1, size-1)\n",
    "        self.obstacles = [(1, 1), (2, 2), (3, 3), (1, 3)]\n",
    "        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up\n",
    "        self.action_names = ['Right', 'Left', 'Down', 'Up']\n",
    "\n",
    "    def reset(self):\n",
    "        self.state = (0, 0)\n",
    "        return self.state\n",
    "\n",
    "    def step(self, action_idx):\n",
    "        action = self.actions[action_idx]\n",
    "        next_state = (self.state[0] + action[0], self.state[1] + action[1])\n",
    "\n",
    "        # Check boundaries\n",
    "        if next_state[0] < 0 or next_state[0] >= self.size or \\\n",
    "           next_state[1] < 0 or next_state[1] >= self.size:\n",
    "            next_state = self.state\n",
    "        \n",
    "        # Check obstacles\n",
    "        if next_state in self.obstacles:\n",
    "            next_state = self.state\n",
    "\n",
    "        self.state = next_state\n",
    "        \n",
    "        reward = -1\n",
    "        done = False\n",
    "        if self.state == self.goal:\n",
    "            reward = 10\n",
    "            done = True\n",
    "        \n",
    "        return next_state, reward, done\n",
    "\n",
    "    def get_all_states(self):\n",
    "        states = []\n",
    "        for i in range(self.size):\n",
    "            for j in range(self.size):\n",
    "                if (i, j) not in self.obstacles:\n",
    "                    states.append((i, j))\n",
    "        return states"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Visualization Helper Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_grid(env, value_function=None, title=\"GridWorld\"):\n",
    "    grid = np.zeros((env.size, env.size))\n",
    "    \n",
    "    # Mark obstacles\n",
    "    for obs in env.obstacles:\n",
    "        grid[obs] = -1\n",
    "    \n",
    "    plt.figure(figsize=(6, 6))\n",
    "    \n",
    "    if value_function is not None:\n",
    "        # Fill grid with values\n",
    "        val_grid = np.full((env.size, env.size), np.nan)\n",
    "        for i in range(env.size):\n",
    "            for j in range(env.size):\n",
    "                if (i, j) not in env.obstacles:\n",
    "                    val_grid[i, j] = value_function.get((i, j), 0)\n",
    "        \n",
    "        sns.heatmap(val_grid, annot=True, fmt=\".2f\", cmap=\"viridis\", cbar=True, mask=np.isnan(val_grid))\n",
    "    else:\n",
    "        # Just show layout\n",
    "        sns.heatmap(grid, annot=False, cbar=False, cmap=\"Greys\", vmin=-1, vmax=0)\n",
    "        \n",
    "        # Annotate Start and Goal\n",
    "        plt.text(0.5, 0.5, 'S', ha='center', va='center', color='green', fontsize=20, fontweight='bold')\n",
    "        plt.text(env.size-0.5, env.size-0.5, 'G', ha='center', va='center', color='red', fontsize=20, fontweight='bold')\n",
    "        \n",
    "        # Annotate Obstacles\n",
    "        for obs in env.obstacles:\n",
    "            plt.text(obs[1]+0.5, obs[0]+0.5, 'X', ha='center', va='center', color='black', fontsize=20)\n",
    "\n",
    "    plt.title(title)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. TD(0) Algorithm\n",
    "\n",
    "The TD(0) update rule is:\n",
    "$$ V(S) \\leftarrow V(S) + \\alpha [R + \\gamma V(S') - V(S)] $$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def td_zero(env, num_episodes, alpha=0.1, gamma=0.9):\n",
    "    # Initialize Value Function\n",
    "    V = {state: 0 for state in env.get_all_states()}\n",
    "    V[env.goal] = 0 # Terminal state value is 0 usually, but here we get reward on transition to it.\n",
    "    # Actually, if we treat goal as terminal, V(goal) should be 0. \n",
    "    # The reward comes when entering the goal.\n",
    "    \n",
    "    # For visualization tracking\n",
    "    value_history = []\n",
    "    \n",
    "    for episode in range(num_episodes):\n",
    "        state = env.reset()\n",
    "        done = False\n",
    "        \n",
    "        while not done:\n",
    "            # Random Policy: Choose random action\n",
    "            action_idx = np.random.choice(len(env.actions))\n",
    "            \n",
    "            next_state, reward, done = env.step(action_idx)\n",
    "            \n",
    "            # TD Update\n",
    "            current_val = V[state]\n",
    "            next_val = V[next_state] if not done else 0\n",
    "            \n",
    "            td_target = reward + gamma * next_val\n",
    "            td_error = td_target - current_val\n",
    "            \n",
    "            V[state] = current_val + alpha * td_error\n",
    "            \n",
    "            state = next_state\n",
    "        \n",
    "        # Store value function snapshot every few episodes\n",
    "        if episode % 10 == 0:\n",
    "            value_history.append(V.copy())\n",
    "            \n",
    "    return V, value_history"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Running the Experiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "env = GridWorld()\n",
    "print(\"Environment Layout:\")\n",
    "plot_grid(env, title=\"GridWorld Layout\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run TD(0)\n",
    "num_episodes = 500\n",
    "alpha = 0.1\n",
    "gamma = 0.9\n",
    "\n",
    "V_final, V_history = td_zero(env, num_episodes, alpha, gamma)\n",
    "\n",
    "print(f\"Training completed for {num_episodes} episodes.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualizing Results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Final Value Function Heatmap:\")\n",
    "plot_grid(env, V_final, title=\"Final Value Function\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize Value Function Evolution for a specific state (e.g., Start State)\n",
    "start_state_values = [v_hist[(0,0)] for v_hist in V_history]\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "plt.plot(range(0, num_episodes, 10), start_state_values)\n",
    "plt.title(\"Value of Start State (0,0) over Episodes\")\n",
    "plt.xlabel(\"Episodes\")\n",
    "plt.ylabel(\"Value V(s)\")\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('u:/Christ/MSc(Ai ML)/Sem 5/RL/Lab/TD0_Visualized.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully.")
