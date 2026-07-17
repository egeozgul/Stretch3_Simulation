import gym
import numpy as np
from .render.game import Game
from gym import spaces
from .items import Tomato, Lettuce, Onion, Peas, Plate, Knife, Delivery, Agent, Food, Blender, Oven, BlendedBowl, Patty
import copy

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "peas", "blender", "oven", "blended_bowl", "patty"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8, "peas": 9, "blender": 10, "oven": 11, "blended_bowl": 12, "patty": 13}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad", "peas salad", "lettuce-peas salad", "lettuce-peas-tomato-patty"]

class Overcooked_V1(gym.Env):

    """
    Overcooked Domain Description
    ------------------------------
    Agent with primitive actions ["right", "down", "left", "up"]
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    
    1) Agent is allowed to pick up/put down food/plate on the counter;
    2) Agent is allowed to chop food into pieces if the food is on the cutting board counter;
    3) Agent is allowed to deliver food to the delivery counter;
    4) Only unchopped food is allowed to be chopped;
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 5
        }

    def __init__(self, grid_dim, task, rewardList, map_type = "A", n_agent = 2, obs_radius = 2, mode = "vector", debug = False):

        """
        Parameters
        ----------
        gird_dim : tuple(int, int)
            The size of the grid world([7, 7]/[9, 9]).
        task : int
            The index of the target recipe.
        rewardList : dictionary
            The list of the reward.
            e.g rewardList = {"subtask finished": 10, "correct delivery": 200, "wrong delivery": -5, "step penalty": -0.1}
        map_type : str 
            The type of the map(A/B/C/D).
        n_agent: int
            The number of the agents.
        obs_radius: int
            The radius of the agents.
        mode: string
            The type of the observation(vector/image).
        debug : bool
            Whehter print the debug information.
        """

        self.xlen, self.ylen = grid_dim
        # Initialize game object for image mode or when debug is enabled
        if debug or mode == "image":
            self.game = Game(self)

        self.task = task
        self.rewardList = rewardList
        self.mapType = map_type
        self.debug = debug
        self.n_agent = n_agent
        self.mode = mode
        self.obs_radius = obs_radius

        map = []

        if self.xlen == 7 and self.ylen == 7:
            if self.n_agent == 2:
                if self.mapType == "A":
                    map =  [[1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 8],
                            [7, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    map =  [[1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 1, 2, 0, 4],
                            [6, 0, 0, 1, 0, 0, 8],
                            [7, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 5],
                            [1, 1, 1, 1, 1, 5, 1]]  
                elif self.mapType == "C":
                    map =  [[1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 1, 2, 0, 4],
                            [6, 0, 0, 1, 0, 0, 8],
                            [7, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 5, 1]]
                elif self.mapType == "D":
                    # Map D: peas instead of onion for lettuce-tomato-pea patty, with blender and two ovens
                    map =  [[1, 1, 1, 11, 11, 3, 1],
                            [6, 0, 2, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 9],
                            [7, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 10, 1, 1, 1]]
            elif self.n_agent == 3:
                if self.mapType == "A":
                    map =  [[1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 8],
                            [7, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 2, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    map =  [[1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 1, 2, 0, 4],
                            [6, 0, 0, 1, 0, 0, 8],
                            [7, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 2, 1, 0, 0, 5],
                            [1, 1, 1, 1, 1, 5, 1]] 
                elif self.mapType == "C":
                    map =  [[1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 1, 2, 0, 4],
                            [6, 0, 0, 1, 0, 0, 8],
                            [7, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 2, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 5, 1]]
                elif self.mapType == "D":
                    # Map D: peas instead of onion for lettuce-tomato-pea patty, with blender and two ovens
                    map =  [[1, 1, 1, 11, 11, 3, 1],
                            [6, 0, 2, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 9],
                            [7, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 2, 0, 0, 0, 5],
                            [1, 1, 1, 10, 1, 1, 1]]
        elif self.xlen == 9 and self.ylen == 9:
            if self.n_agent == 2:
                if self.mapType == "A":
                    map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 0, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 0, 0, 8],
                            [7, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 1, 0, 2, 0, 4],
                            [6, 0, 0, 0, 1, 0, 0, 0, 8],
                            [7, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1, 5, 1]]
                elif self.mapType == "C":
                    map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 1, 0, 2, 0, 4],
                            [6, 0, 0, 0, 1, 0, 0, 0, 8],
                            [7, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1, 5, 1]]
                elif self.mapType == "D":
                    # Map D: peas instead of onion for lettuce-tomato-pea patty, with blender and two ovens
                    map =  [[1, 1, 1, 1, 11, 11, 1, 3, 1],
                            [6, 0, 2, 0, 0, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 0, 0, 9],
                            [7, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 10, 1, 1, 1, 1]]
            elif self.n_agent == 3:
                if self.mapType == "A":
                    map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 0, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 0, 0, 8],
                            [7, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 2, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 1, 0, 2, 0, 4],
                            [6, 0, 0, 0, 1, 0, 0, 0, 8],
                            [7, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 2, 0, 1, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1, 5, 1]]
                elif self.mapType == "C":
                    map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                            [6, 0, 2, 0, 1, 0, 2, 0, 4],
                            [6, 0, 0, 0, 1, 0, 0, 0, 8],
                            [7, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 2, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 1, 1, 1, 5, 1]]
                elif self.mapType == "D":
                    # Map D: peas instead of onion for lettuce-tomato-pea patty, with blender and two ovens
                    map =  [[1, 1, 1, 1, 11, 11, 1, 3, 1],
                            [6, 0, 2, 0, 0, 0, 2, 0, 4],
                            [6, 0, 0, 0, 0, 0, 0, 0, 9],
                            [7, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 2, 0, 0, 0, 0, 0, 5],
                            [1, 1, 1, 1, 10, 1, 1, 1, 1]]
        self.initMap = map
        self.map = [row[:] for row in self.initMap]
        # O(1) color -> agent index lookup (replaces AGENTCOLOR.index in step loop).
        self._color_to_idx = {c: i for i, c in enumerate(AGENTCOLOR)}

        # Cache pomap templates (created once, copied when needed)
        self._pomap_templates = self._create_pomap_templates()
        
        # Position-indexed item lookup for O(1) access
        self._position_to_item = {}
        
        self.oneHotTask = []
        for t in TASKLIST:
            if t == self.task:
                self.oneHotTask.append(1)
            else:
                self.oneHotTask.append(0)

        self._createItems()
        
        # DEBUG: Check if agent count matches expected
        actual_agent_count = len(self.agent)
        if actual_agent_count != n_agent:
            print(f"WARNING: Expected {n_agent} agents but map has {actual_agent_count} agents!")
            print(f"  Map type: {map_type}, Grid: {grid_dim}")
        
        self.n_agent = len(self.agent)

        #action: move(up, down, left, right), stay
        self.action_space = spaces.Discrete(5)

        #Observation: agent(pos[x,y]) dim = 2
        #    knife(pos[x,y]) dim = 2
        #    delivery (pos[x,y]) dim = 2
        #    plate(pos[x,y]) dim = 2
        #    food(pos[x,y]/status) dim = 3

        self._initObs()
        obs = self._get_obs()
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(obs[0]),), dtype=np.float32)


    def _create_pomap_templates(self):
        """Create cached pomap templates based on grid size and map type."""
        templates = {}
        if self.xlen == 7 and self.ylen == 7:
            templates["A"] = [[1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1]]
            templates["B"] = [[1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1]]
            templates["C"] = [[1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1]]
            templates["D"] = [[1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1]]
        elif self.xlen == 9 and self.ylen == 9:
            templates["A"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]]
            templates["B"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]]
            templates["C"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]]
            templates["D"] = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]]
        # Keep as list-of-lists: per-agent pomap is accessed with Python indexing
        # (agent.pomap[x][y]) in the A* inner loop, which is faster on lists than
        # numpy scalar access. Copy is done row-by-row in _get_vector_obs.
        return templates

    def _createItems(self):
        self.agent = []
        self.knife = []
        self.delivery = []
        self.tomato = []
        self.lettuce = []
        self.onion = []
        self.peas = []
        self.plate = []
        self.blender = []
        self.oven = []
        self.itemList = []
        agent_idx = 0
        for x in range(self.xlen):
            for y in range(self.ylen):
                if self.map[x][y] == ITEMIDX["agent"]:
                    self.agent.append(Agent(x, y, color = AGENTCOLOR[agent_idx]))
                    agent_idx += 1
                elif self.map[x][y] == ITEMIDX["knife"]:
                    self.knife.append(Knife(x, y))
                elif self.map[x][y] == ITEMIDX["delivery"]:
                    self.delivery.append(Delivery(x, y))                    
                elif self.map[x][y] == ITEMIDX["tomato"]:
                    self.tomato.append(Tomato(x, y))
                elif self.map[x][y] == ITEMIDX["lettuce"]:
                    self.lettuce.append(Lettuce(x, y))
                elif self.map[x][y] == ITEMIDX["onion"]:
                    self.onion.append(Onion(x, y))
                elif self.map[x][y] == ITEMIDX["peas"]:
                    self.peas.append(Peas(x, y))
                elif self.map[x][y] == ITEMIDX["plate"]:
                    self.plate.append(Plate(x, y))
                elif self.map[x][y] == ITEMIDX["blender"]:
                    self.blender.append(Blender(x, y))
                elif self.map[x][y] == ITEMIDX["oven"]:
                    self.oven.append(Oven(x, y))
        
        self.itemDic = {"tomato": self.tomato, "lettuce": self.lettuce, "onion": self.onion, "peas": self.peas, "plate": self.plate, "knife": self.knife, "delivery": self.delivery, "agent": self.agent, "blender": self.blender, "oven": self.oven}
        for key in self.itemDic:
            self.itemList += self.itemDic[key]

        self._update_position_lookup()

        self._is_food = [isinstance(item, Food) for item in self.itemList]

        # Per-item ITEMIDX (avoids dict lookup inside the per-step obs loop).
        self._item_idx = [ITEMIDX[item.rawName] if hasattr(item, 'rawName') and item.rawName else ITEMIDX["agent"] for item in self.itemList]
        # Per-item 1/required_chopped_times (only meaningful for Food).
        self._inv_required_chop = [
            1.0 / item.required_chopped_times if self._is_food[i] else 0.0
            for i, item in enumerate(self.itemList)
        ]

        self._obs_size = sum(3 if is_food else 2 for is_food in self._is_food) + len(TASKLIST)

        # Inverse grid dims cached as floats so obs normalisation is a multiply, not a divide.
        self._inv_xlen = 1.0 / self.xlen
        self._inv_ylen = 1.0 / self.ylen

        # Preallocated task one-hot tail (appended to the end of every obs — never changes per-episode).
        self._task_tail = np.zeros(len(TASKLIST), dtype=np.float32)
        for i, t in enumerate(TASKLIST):
            if t == self.task:
                self._task_tail[i] = 1.0


    def _initObs(self):
        # Preallocate observation array
        obs = np.zeros(self._obs_size, dtype=np.float32)
        idx = 0
        for i, item in enumerate(self.itemList):
            obs[idx] = item.x / self.xlen
            obs[idx + 1] = item.y / self.ylen
            idx += 2
            if self._is_food[i]:
                obs[idx] = item.cur_chopped_times / item.required_chopped_times
                idx += 1
        # Add one-hot task encoding
        for i, t in enumerate(TASKLIST):
            obs[idx + i] = 1.0 if t == self.task else 0.0

        for agent in self.agent:
            agent.obs = obs.tolist()  # Convert to list for compatibility
        return [obs.copy() for _ in range(self.n_agent)]


    def _get_vector_state(self):
        # Preallocate state array
        state = np.zeros(self._obs_size, dtype=np.float32)
        idx = 0
        for i, item in enumerate(self.itemList):
            state[idx] = item.x / self.xlen
            state[idx + 1] = item.y / self.ylen
            idx += 2
            if self._is_food[i]:
                state[idx] = item.cur_chopped_times / item.required_chopped_times
                idx += 1
        # Add one-hot task encoding
        for i, t in enumerate(TASKLIST):
            state[idx + i] = 1.0 if t == self.task else 0.0
        return [state.copy() for _ in range(self.n_agent)]

    def _get_image_state(self):
        if self.game is None:
            raise RuntimeError("Cannot get image observations when game is not initialized. Use mode='image' or debug=True.")
        return [self.game.get_image_obs()] * self.n_agent

    def _get_obs(self):
        """
        Returns
        -------
        obs : list
            observation for each agent.
        """

        vec_obs = self._get_vector_obs()
        if self.obs_radius > 0:
            if self.mode == "vector":
                return vec_obs
            elif self.mode == "image":
                return self._get_image_obs()
        else:
            if self.mode == "vector":
                return self._get_vector_state()
            elif self.mode == "image":
                return self._get_image_state()
        return []

    def _get_vector_obs(self):

        """
        Returns
        -------
        vector_obs : list
            vector observation for each agent.
        """

        po_obs = []
        template = self._pomap_templates.get(self.mapType)
        obs_radius = self.obs_radius
        inv_x = self._inv_xlen
        inv_y = self._inv_ylen
        xlen = self.xlen
        ylen = self.ylen
        item_list = self.itemList
        is_food_tbl = self._is_food
        item_idx_tbl = self._item_idx
        inv_chop_tbl = self._inv_required_chop
        task_tail = self._task_tail
        task_tail_len = task_tail.size
        agent_idx_val = ITEMIDX["agent"]
        obs_size = self._obs_size
        task_head = obs_size - task_tail_len
        n_items = len(item_list)

        for agent in self.agent:
            obs = np.zeros(obs_size, dtype=np.float32)
            obs_idx = 0
            agent_obs_idx = 0
            ax = agent.x
            ay = agent.y
            lo_x = ax - obs_radius
            hi_x = ax + obs_radius
            lo_y = ay - obs_radius
            hi_y = ay + obs_radius
            prev_obs = agent.obs

            if template is not None:
                # Reuse the agent's existing pomap rows where possible — saves xlen+1
                # list allocations per agent per step compared to [row[:] for row in template].
                pomap = agent.pomap
                if pomap is None or len(pomap) != xlen:
                    pomap = [row[:] for row in template]
                else:
                    for r in range(xlen):
                        pomap[r][:] = template[r]
            else:
                pomap = [[1] * ylen for _ in range(xlen)]

            for i in range(n_items):
                item = item_list[i]
                is_food = is_food_tbl[i]
                ix = item.x
                iy = item.y
                if obs_radius == 0 or (lo_x <= ix <= hi_x and lo_y <= iy <= hi_y):
                    obs[obs_idx] = ix * inv_x
                    obs[obs_idx + 1] = iy * inv_y
                    obs_idx += 2
                    agent_obs_idx += 2
                    if is_food:
                        obs[obs_idx] = item.cur_chopped_times * inv_chop_tbl[i]
                        obs_idx += 1
                        agent_obs_idx += 1
                    px, py = ix, iy
                else:
                    # Recover integer grid coordinates from previous step's normalised obs,
                    # avoiding the round-trip `int(x*xlen)` of the original implementation.
                    sx = prev_obs[agent_obs_idx] * xlen
                    sy = prev_obs[agent_obs_idx + 1] * ylen
                    if lo_x <= sx <= hi_x and lo_y <= sy <= hi_y:
                        px = item.initial_x
                        py = item.initial_y
                    else:
                        px = int(sx)
                        py = int(sy)
                    obs[obs_idx] = px * inv_x
                    obs[obs_idx + 1] = py * inv_y
                    obs_idx += 2
                    agent_obs_idx += 2
                    if is_food:
                        # Preserve original (possibly buggy) semantics that read
                        # agent.obs[agent_obs_idx - 1].
                        obs[obs_idx] = prev_obs[agent_obs_idx - 1] * inv_chop_tbl[i] if agent_obs_idx > 0 else 0.0
                        obs_idx += 1
                        agent_obs_idx += 1

                pomap[px][py] = item_idx_tbl[i]

            pomap[ax][ay] = agent_idx_val
            agent.pomap = pomap
            # Invalidate cached pomap hash — pomap has just been rebuilt this step.
            agent._pomap_hash = None

            obs[task_head:task_head + task_tail_len] = task_tail
            agent.obs = obs.tolist()
            po_obs.append(obs.copy())
        return po_obs

    def _get_image_obs(self):

        """
        Returns
        -------
        image_obs : list
            image observation for each agent.
        """

        po_obs = []
        frame = self.game.get_image_obs()
        old_image_width, old_image_height, channels = frame.shape
        new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
        new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
        color = (0,0,0)
        obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame

        for idx, agent in enumerate(self.agent):
            agent_obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)
            po_obs.append(agent_obs)
        return po_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def _update_position_lookup(self):
        """Rebuild position-indexed item lookup dictionary."""
        # Note: Position caching disabled - items move during gameplay
        # and keeping cache in sync adds complexity. Linear search is fast enough.
        pass

    def _findItem(self, x, y, itemName):
        """Find item by position and type (linear search)."""
        for item in self.itemDic[itemName]:
            if item.x == x and item.y == y:
                return item
        return None

    @property
    def state_size(self):
        return len(self._get_vector_state()[0])

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    @property
    def action_spaces(self):
        return [self.action_space] * self.n_agent

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def _is_correct_plating(self, item):
        # Patty task: only the cooked Patty is the correct plate target — the
        # raw chopped lettuce/peas/tomato should go to the blender, not a
        # plate, so don't pay the subtask reward for plating those.
        if self.task == "lettuce-peas-tomato-patty":
            return isinstance(item, Patty)
        # Salad tasks: reward each recipe ingredient as it lands on the plate.
        return getattr(item, "rawName", "") in self.task

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    def reset(self):

        """
        Returns
        -------
        obs : list
            observation for each agent.
        """

        self.map = [row[:] for row in self.initMap]
        self._createItems()
        self._initObs()
        if self.debug and self.game is not None:
            self.game.on_cleanup()

        return self._get_obs()
    
    def step(self, action):

        """
        Parameters
        ----------
        action: list
            action for each agent

        Returns
        -------
        obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """

        reward = self.rewardList["step penalty"]
        done = False
        info = {}
        info['cur_mac'] = action
        info['mac_done'] = [True] * self.n_agent
        info['collision'] = []

        all_action_done = False

        for agent in self.agent:
            agent.moved = False

        # Debug printing is now handled at the macro-action level in overcooked_MA_V1.py
        # if self.debug:
        #     print("in overcooked primitive actions:", action)

        while not all_action_done:
            for idx, agent in enumerate(self.agent):
                agent_action = action[idx]
                if agent.moved:
                    continue
                agent.moved = True

                if agent_action < 4:
                    target_x = agent.x + DIRECTION[agent_action][0]
                    target_y = agent.y + DIRECTION[agent_action][1]
                    target_name = ITEMNAME[self.map[target_x][target_y]]

                    if target_name == "agent":
                        target_agent = self._findItem(target_x, target_y, target_name)
                        if target_agent is not None and not target_agent.moved:
                            agent.moved = False
                            target_agent_action = action[self._color_to_idx[target_agent.color]]
                            if target_agent_action < 4:
                                new_target_agent_x = target_agent.x + DIRECTION[target_agent_action][0]
                                new_target_agent_y = target_agent.y + DIRECTION[target_agent_action][1]
                                if new_target_agent_x == agent.x and new_target_agent_y == agent.y:
                                    target_agent.move(new_target_agent_x, new_target_agent_y)
                                    agent.move(target_x, target_y)
                                    agent.moved = True
                                    target_agent.moved = True
                    elif  target_name == "space":
                        self.map[agent.x][agent.y] = ITEMIDX["space"]
                        agent.move(target_x, target_y)
                        self.map[target_x][target_y] = ITEMIDX["agent"]
                    #pickup and chop
                    elif not agent.holding:
                        if target_name == "tomato" or target_name == "lettuce" or target_name == "plate" or target_name == "onion" or target_name == "peas":
                            item = self._findItem(target_x, target_y, target_name)
                            if item is not None:
                                agent.pickup(item)
                                self.map[target_x][target_y] = ITEMIDX["counter"]
                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            if knife is None:
                                pass
                            elif isinstance(knife.holding, Plate):
                                item = knife.holding
                                knife.release()
                                agent.pickup(item)
                            elif isinstance(knife.holding, Food):
                                # BlendedBowl is now a Food with chopped=True, so it will be picked up
                                if knife.holding.chopped:
                                    item = knife.holding
                                    knife.release()
                                    agent.pickup(item)
                                else:
                                    knife.holding.chop()
                                    if knife.holding.chopped:
                                        if knife.holding.rawName in self.task:
                                            # For the patty task, peas go directly into the
                                            # blender and do NOT need chopping. Rewarding
                                            # chop-peas via a substring match encourages
                                            # wasted effort, so suppress it here.
                                            if not (self.task == "lettuce-peas-tomato-patty"
                                                    and knife.holding.rawName == "peas"):
                                                # Pay chop credit at most once per food per
                                                # episode. Wrong-delivery resets the food to
                                                # unchopped state, but the flag persists so
                                                # the agent can't re-earn this reward.
                                                if not knife.holding._chop_rewarded:
                                                    reward += self.rewardList["subtask finished"]
                                                    knife.holding._chop_rewarded = True
                        elif target_name == "blender":
                            blender = self._findItem(target_x, target_y, target_name)
                            if blender is None:
                                pass
                            elif blender.blended:
                                # Pick up BlendedBowl from blender
                                bowl = blender.create_blended_bowl()
                                if bowl:
                                    agent.pickup(bowl)
                            elif blender.can_blend() and not blender.blended:
                                # Perform one step of blending (takes 5 steps total)
                                if blender.blend_step():
                                    if not getattr(blender, "_blend_rewarded", False):
                                        reward += self.rewardList["subtask finished"]
                                        blender._blend_rewarded = True
                        elif target_name == "oven":
                            oven = self._findItem(target_x, target_y, target_name)
                            if oven is None:
                                pass
                            elif oven.cooked:
                                # Create a Patty from the cooked contents
                                patty = oven.create_patty()
                                if patty:
                                    agent.pickup(patty)
                                    # Reward bridge: closes the largest gap in
                                    # the patty-task reward valley (between
                                    # cook-complete and final delivery). Same
                                    # magnitude as a chop/blend/cook subtask.
                                    reward += self.rewardList["subtask finished"]
                            elif oven.cooking and not oven.cooked:
                                # Perform one step of cooking (takes 10 steps total)
                                if oven.cook_step():
                                    if not getattr(oven, "_cook_rewarded", False):
                                        reward += self.rewardList["subtask finished"]
                                        oven._cook_rewarded = True
                    #put down
                    elif agent.holding:
                        if target_name == "counter":
                            if agent.holding.rawName in ["tomato", "lettuce", "onion", "peas", "plate"]:
                                self.map[target_x][target_y] = ITEMIDX[agent.holding.rawName]
                            agent.putdown(target_x, target_y)
                        elif target_name == "plate":
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    plate = self._findItem(target_x, target_y, target_name)
                                    if plate is not None:
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        plate.contain(item)
                                        # Reward bridge: closes the second gap
                                        # (patty pickup -> plate). Only fires
                                        # when a Patty is plated, so other
                                        # tasks are unaffected.
                                        if isinstance(item, Patty):
                                            reward += self.rewardList["subtask finished"]

                        elif target_name == "knife":
                            knife = self._findItem(target_x, target_y, target_name)
                            if knife is None:
                                pass
                            elif not knife.holding:
                                # Don't allow finished items (Patty, BlendedBowl) on the knife
                                if isinstance(agent.holding, (Patty, BlendedBowl)):
                                    pass
                                else:
                                    item = agent.holding
                                    agent.putdown(target_x, target_y)
                                    knife.hold(item)
                            elif isinstance(knife.holding, Food) and isinstance(agent.holding, Plate):
                                item = knife.holding
                                if item.chopped:
                                    knife.release()
                                    agent.holding.contain(item)
                                    if self._is_correct_plating(item) and not getattr(item, "_plate_rewarded", False):
                                        reward += self.rewardList["subtask finished"]
                                        item._plate_rewarded = True
                            elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Food):
                                plate_item = knife.holding
                                food_item = agent.holding
                                if food_item.chopped:
                                    knife.release()
                                    agent.pickup(plate_item)
                                    if isinstance(agent.holding, Plate):
                                        agent.holding.contain(food_item)
                                        if self._is_correct_plating(food_item) and not getattr(food_item, "_plate_rewarded", False):
                                            reward += self.rewardList["subtask finished"]
                                            food_item._plate_rewarded = True
                        elif target_name == "blender":
                            blender = self._findItem(target_x, target_y, target_name)
                            if blender is not None and isinstance(agent.holding, Food) and not isinstance(agent.holding, BlendedBowl) and not isinstance(agent.holding, Patty) and not blender.blended:
                                # Add food to blender (lettuce/tomato must be chopped first)
                                # BlendedBowl and Patty cannot be put into the blender
                                food = agent.holding
                                if blender.add_food(food):
                                    agent.holding = None
                            # Note: blended contents must be transferred to oven, not picked up directly
                        elif target_name == "oven":
                            oven = self._findItem(target_x, target_y, target_name)
                            if oven is None:
                                pass
                            elif isinstance(agent.holding, BlendedBowl) and not oven.cooking and not oven.cooked:
                                # Put BlendedBowl into oven to start cooking
                                bowl = agent.holding
                                agent.holding = None
                                oven.add_blended_bowl(bowl)
                                # Reward bridge: closes the blend->cook gap so
                                # agents that successfully blended don't stall
                                # on the way to the oven.
                                reward += self.rewardList["subtask finished"]
                            elif isinstance(agent.holding, Patty):
                                # Can't put patty into oven
                                pass
                        elif target_name == "delivery":
                            if isinstance(agent.holding, Plate):
                                if agent.holding.containing:
                                    dishName = ""
                                    foodList = [Lettuce, Onion, Peas, Tomato]
                                    foodNames = ["lettuce", "onion", "peas", "tomato"]  # Corresponding raw names
                                    foodInPlate = [-1] * len(foodList)
                                    has_blender_blended = False
                                    has_patty = False
                                    
                                    # Check if plate contains a BlendedBowl or Patty
                                    for f in agent.holding.containing:
                                        if isinstance(f, BlendedBowl):
                                            has_blender_blended = True
                                            break
                                        if isinstance(f, Patty):
                                            has_patty = True
                                            # Patty always contains all three ingredients
                                            # (same as a salad — assume lettuce, peas, tomato)
                                            foodInPlate[0] = -2  # lettuce
                                            foodInPlate[2] = -2  # peas
                                            foodInPlate[3] = -2  # tomato
                                            break
                                    
                                    # If no patty, check regular food items
                                    if not has_patty:
                                        for f in range(len(agent.holding.containing)):
                                            for i in range(len(foodList)):
                                                if isinstance(agent.holding.containing[f], foodList[i]):
                                                    foodInPlate[i] = f
                                    
                                    for i in range(len(foodList)):
                                        if foodInPlate[i] > -1:
                                            dishName += agent.holding.containing[foodInPlate[i]].rawName + "-"
                                        elif foodInPlate[i] == -2:  # From patty
                                            dishName += foodNames[i] + "-"
                                    
                                    # Add patty suffix if patty is in the plate
                                    if has_patty:
                                        dishName = dishName[:-1] + "-patty" if dishName else "patty"
                                    elif dishName:
                                        dishName = dishName[:-1] + " salad"
                                    
                                    # Check if blenderBlended was delivered instead of patty
                                    if has_blender_blended and self.task == "lettuce-peas-tomato-patty":
                                        # BlendedBowl is NOT the final product for a patty task.
                                        # Previously: reward += 5 and done = True, which created a
                                        # local-optima trap (episode terminated on a suboptimal +5
                                        # instead of the +200 patty delivery). Now: treat exactly
                                        # like any wrong delivery so agents keep learning the full
                                        # cook->patty->plate->deliver chain.
                                        reward += self.rewardList["wrong delivery"]
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        food = item.containing
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        if food is not None:
                                            for f in food:
                                                f.refresh()
                                                self.map[f.x][f.y] = ITEMIDX[f.rawName]
                                        info['blender_instead_of_patty'] = True
                                    elif dishName == self.task:
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        self.delivery[0].hold(item)
                                        reward += self.rewardList["correct delivery"]
                                        done = True
                                    else:
                                        reward += self.rewardList["wrong delivery"]
                                        item = agent.holding
                                        agent.putdown(target_x, target_y)
                                        food = item.containing
                                        item.release()
                                        item.refresh()
                                        self.map[item.x][item.y] = ITEMIDX[item.name]
                                        if food is not None:
                                            for f in food:
                                                f.refresh()
                                                self.map[f.x][f.y] = ITEMIDX[f.rawName]
                                else:
                                    reward += self.rewardList["wrong delivery"]
                                    plate = agent.holding
                                    agent.putdown(target_x, target_y)
                                    plate.refresh()
                                    self.map[plate.x][plate.y] = ITEMIDX[plate.name]
                            else:
                                reward += self.rewardList["wrong delivery"]
                                food = agent.holding
                                agent.putdown(target_x, target_y)
                                food.refresh()
                                self.map[food.x][food.y] = ITEMIDX[food.rawName]

                        elif target_name in ["tomato", "lettuce", "onion", "peas"]:
                            item = self._findItem(target_x, target_y, target_name)
                            if item is not None and item.chopped and isinstance(agent.holding, Plate):
                                agent.holding.contain(item)
                                self.map[target_x][target_y] = ITEMIDX["counter"]

            all_action_done = True
            for agent in self.agent:
                if agent.moved == False:
                    all_action_done = False
        
        # Auto-cook: oven automatically cooks each timestep when it has contents.
        # The outer `not oven.cooked` guard prevents the cook_step() early-return
        # branch (which returns True every call once cooked), but the reward is
        # also gated on _cook_rewarded so a wrong-delivery → re-cook cycle can't
        # re-earn this credit either.
        for oven in self.oven:
            if oven.cooking and not oven.cooked:
                if oven.cook_step():
                    if not getattr(oven, "_cook_rewarded", False):
                        reward += self.rewardList["subtask finished"]
                        oven._cook_rewarded = True
        
        return self._get_obs(), [reward] * self.n_agent, done, info

    def render(self, mode='human'):
        if self.game is None:
            raise RuntimeError("Cannot render when game is not initialized. Enable debug mode or use image observations.")
        return self.game.on_render()

    





