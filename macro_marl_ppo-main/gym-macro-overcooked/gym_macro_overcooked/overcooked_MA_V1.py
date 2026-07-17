import heapq
import itertools
import numpy as np
from gym import spaces
from .items import Tomato, Onion, Lettuce, Peas, Plate, Knife, Delivery, Agent, Food, Blender, Oven, BlendedBowl, Patty
from .overcooked_V1 import Overcooked_V1, TASKLIST
from .mac_agent import MacAgent
import random

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "peas", "blender", "oven", "blended_bowl", "patty"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8, "peas": 9, "blender": 10, "oven": 11, "blended_bowl": 12, "patty": 13}
AGENTCOLOR = ["blue", "magenta", "green", "yellow"]
ACTIONIDX = {"right": 0, "down": 1, "left": 2, "up": 3, "stay": 4}
PRIMITIVEACTION =["right", "down", "left", "up", "stay"]

class AStarAgent(object):
    __slots__ = ("x", "y", "g", "dis", "action", "pass_agent")

    def __init__(self, x, y, g, dis, action, pass_agent):

        """
        Parameters
        ----------
        x : int
            X position of the agent.
        y : int
            Y position of the agent.
        g : int
            Cost of the path from the start node to n.
        dis : int
            Distance of the current path.
            g + h
        pass_agent : int
            Whether there is other agent in the path.
        """

        self.x = x
        self.y = y
        self.g = g
        self.dis = dis
        self.action = action
        self.pass_agent = pass_agent

    def __lt__(self, other):
        if self.dis != other.dis:
            return self.dis <= other.dis
        else:
            return self.pass_agent <= other.pass_agent

class Overcooked_MA_V1(Overcooked_V1):

    """
    Overcooked Domain Description
    ------------------------------
    ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "peas", "blender", "oven"]
    map_type = ["A", "B", "C", "D"]

    Only macro-action is available in this env.
    Macro-actions in map A/D:
    ["stay", "get tomato", "get lettuce", "get onion", "get peas", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "right", "down", "left", "up"]
    Macro-actions in map B/C:
    ["stay", "get tomato", "get lettuce", "get onion", "get peas", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter", "right", "down", "left", "up"]
    
    1) Agent is allowed to pick up/put down food/plate on the counter;
    2) Agent is allowed to chop food into pieces if the food is on the cutting board counter;
    3) Agent is allowed to deliver food to the delivery counter;
    4) Only unchopped food is allowed to be chopped;
    """
        
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
            The type of the map(A/B/C).
        n_agent: int
            The number of the agents.
        obs_radius: int
            The radius of the agents.
        mode: string
            The type of the observation(vector/image).
        debug : bool
            Whehter print the debug information.
        """

        super().__init__(grid_dim, task, rewardList, map_type, n_agent, obs_radius, mode, debug)
        self.macroAgent = []
        self._createMacroAgents()
        self.macroActionItemList = []
        self._createMacroActionItemList()
        
        # A* path cache: (agent_x, agent_y, target_x, target_y, pomap_hash) -> first_action
        self._astar_cache = {}
        self._astar_cache_hits = 0
        self._astar_cache_misses = 0

        if map_type == "A":
            # Keep "get plate 2" in action list for backwards compatibility with trained policies
            # Even though Map A only has 1 plate, this prevents action index misalignment
            self.macroActionName = ["stay", "get tomato", "get lettuce", "get onion", "get peas", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "right", "down", "left", "up"]
        elif map_type == "D":
            # Map D has blender and two ovens - add blend and cook macro-actions
            # "get patty" is at the end to maintain backwards compatibility with policies trained before it was added
            self.macroActionName = ["stay", "get tomato", "get lettuce", "get onion", "get peas", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to blender", "blend", "get blended bowl", "go to oven 1", "go to oven 2", "cook", "get patty", "right", "down", "left", "up"]
        else:
            self.macroActionName = ["stay", "get tomato", "get lettuce", "get onion", "get peas", "get plate 1", "get plate 2", "go to knife 1", "go to knife 2", "deliver", "chop", "go to counter", "right", "down", "left", "up"]
        self.action_space = spaces.Discrete(len(self.macroActionName))
        # Cache the "get plate 1" index — used by the observation-based fallback
        # path in _findPOitem on every call to avoid an O(n) list.index().
        self._plate_1_idx = self.macroActionName.index("get plate 1")
        # Cache "right" index — this is the first primitive-direction macro-action
        # and is used in the main branch of _computeLowLevelActions every step.
        self._right_idx = self.macroActionName.index("right")
        self._stay_action_idx = ACTIONIDX["stay"]

        if self.xlen == 7 and self.ylen == 7:
            if self.mapType == "B":
                self.counterSequence = [3, 2, 4, 1, 5]
            elif self.mapType == "C":
                self.counterSequence = [3, 2, 4, 1]
        elif self.xlen == 9 and self.ylen == 9:
            if self.mapType == "B":
                self.counterSequence = [4, 3, 5, 2, 6, 1, 7]
            elif self.mapType == "C":
                self.counterSequence = [4, 3, 5, 2, 6, 1]

    def _createMacroAgents(self):
        for agent in self.agent:
            self.macroAgent.append(MacAgent())

    def _createMacroActionItemList(self):
        self.macroActionItemList = []
        for key in self.itemDic:
            if key != "agent":
                self.macroActionItemList += self.itemDic[key]

    def macro_action_sample(self):
        macro_actions = []
        for agent in self.agent:
            macro_actions.append(random.randint(0, self.action_space.n - 1))
        return macro_actions     

    def build_agents(self):
        raise

    def build_macro_actions(self):
        raise

    def _findPOitem(self, agent, macro_action):
    
        """
        Parameters
        ----------
        agent : Agent
        macro_action: int

        Returns
        -------
        x : int
            X position of the item in the observation of the agent.
        y : int
            Y position of the item in the observation of the agent.
        """
        
        # Handle actions directly using item lists (more reliable than observation indices)
        # This approach is robust to maps with different item configurations (e.g., Map A has no peas)
        action_name = self.macroActionName[macro_action]
        
        # Knife actions
        if action_name == "go to knife 1" and len(self.knife) >= 1:
            return self.knife[0].x, self.knife[0].y
        elif action_name == "go to knife 2" and len(self.knife) >= 2:
            return self.knife[1].x, self.knife[1].y
        
        # Plate actions - directly use plate list positions
        elif action_name == "get plate 1" and len(self.plate) >= 1:
            return self.plate[0].x, self.plate[0].y
        elif action_name == "get plate 2" and len(self.plate) >= 2:
            return self.plate[1].x, self.plate[1].y
        
        # Delivery action
        elif action_name == "deliver" and len(self.delivery) >= 1:
            return self.delivery[0].x, self.delivery[0].y
        
        # Food actions - directly use food list positions
        elif action_name == "get tomato" and len(self.tomato) >= 1:
            return self.tomato[0].x, self.tomato[0].y
        elif action_name == "get lettuce" and len(self.lettuce) >= 1:
            return self.lettuce[0].x, self.lettuce[0].y
        elif action_name == "get onion" and len(self.onion) >= 1:
            return self.onion[0].x, self.onion[0].y
        elif action_name == "get peas" and len(self.peas) >= 1:
            return self.peas[0].x, self.peas[0].y
        
        # Blended bowl - search dynamically as it's created during gameplay
        # The bowl can be: 1) at the blender (blender.blended=True), 2) on a knife, 3) held by an agent
        elif action_name == "get blended bowl":
            min_dist = float('inf')
            target_x, target_y = None, None
            
            # Check blender for a ready blended bowl
            if self.blender:
                for blender in self.blender:
                    if blender.blended:
                        dist = self._calDistance(agent.x, agent.y, blender.x, blender.y)
                        if dist < min_dist:
                            min_dist = dist
                            target_x, target_y = blender.x, blender.y
            
            # Check knives for a blended bowl
            for knife in self.knife:
                if knife.holding and isinstance(knife.holding, BlendedBowl):
                    dist = self._calDistance(agent.x, agent.y, knife.x, knife.y)
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = knife.x, knife.y
            
            if target_x is not None:
                return target_x, target_y
            # If not found, return agent's position (will cause immediate completion)
            return agent.x, agent.y
        
        # Patty - search dynamically as it's created during gameplay
        # The patty is at the oven when cooked
        elif action_name == "get patty":
            min_dist = float('inf')
            target_x, target_y = None, None
            
            # Check ovens for a ready patty
            if self.oven:
                for oven in self.oven:
                    if oven.cooked:
                        dist = self._calDistance(agent.x, agent.y, oven.x, oven.y)
                        if dist < min_dist:
                            min_dist = dist
                            target_x, target_y = oven.x, oven.y
            
            if target_x is not None:
                return target_x, target_y
            # If not found, return far away position so agent navigates somewhere neutral
            # This prevents infinite loop when patty isn't available yet
            return 0, 0
        
        # Fallback to observation-based calculation for any other actions
        # (this maintains backwards compatibility)
        foodIdx = self._plate_1_idx
        if macro_action < foodIdx:
            idx = (macro_action - 1) * 3
        else:
            idx = (macro_action - 1) * 2 + (foodIdx - 1)
        return int(agent.obs[idx] * self.xlen), int(agent.obs[idx + 1] * self.ylen)

    def reset(self):
                
        """
        Returns
        -------
        macro_obs : list
            observation for each agent.
        """

        super().reset()
        for agent in self.macroAgent:
            agent.reset()
        # Clear A* cache on reset since map state changes
        self._astar_cache.clear()
        return self._get_macro_obs()

    def run(self, macro_actions):

        """
        Parameters
        ----------
        macro_actions: list
            macro_action for each agent

        Returns
        -------
        macro_obs : list
            observation for each agent.
        rewards : list
        terminate : list
        info : dictionary
        """

        if self.debug:
            # Print macro-actions being executed
            macro_action_names = []
            for idx, agent in enumerate(self.agent):
                if self.macroAgent[idx].cur_macro_action_done:
                    # Agent is sampling a new macro-action
                    macro_action_names.append(f"Agent {idx}: {self.macroActionName[macro_actions[idx]]}")
                else:
                    # Agent is continuing previous macro-action
                    macro_action_names.append(f"Agent {idx}: {self.macroActionName[self.macroAgent[idx].cur_macro_action]} (continuing)")
            print("Macro-actions:", " | ".join(macro_action_names))

        actions = self._computeLowLevelActions(macro_actions)
        
        obs, rewards, terminate, info = self.step(actions)

        self._checkMacroActionDone()
        self._checkCollision(info)
        cur_mac = self._collectCurMacroActions()
        mac_done = self._computeMacroActionDone()

        self._createMacroActionItemList()

        info = {'cur_mac': cur_mac, 'mac_done': mac_done}
        return  self._get_macro_obs(), rewards, terminate, info

    def _checkCollision(self, info):
        for idx in info["collision"]:
            self.macroAgent[idx].cur_macro_action_done = True

    def _checkMacroActionDone(self):
        macroActionName = self.macroActionName
        mapType = self.mapType
        # Cache: which item types are still REACHABLE this tick? An item is
        # reachable if any of:
        #   1. It has a map cell (sitting on a counter or original spawn).
        #   2. It is held by a knife (chopped or being chopped — the agent
        #      can navigate to the knife and pick it up once chopped).
        #   3. It is held by an agent (transient; the holder may drop it).
        # An item is UNREACHABLE if it has been added to a blender or
        # encapsulated inside a BlendedBowl/Patty — those have no way back.
        # The macro state machine has no done-condition for "target gone"
        # so without this short-circuit the agent stalls on "(continuing)"
        # forever after blending consumes the raw foods.
        map_arr = np.asarray(self.map)
        from .items import Tomato, Lettuce, Onion, Peas, Plate as _Plate
        _classes = {"tomato": Tomato, "lettuce": Lettuce,
                    "onion": Onion, "peas": Peas, "plate": _Plate}

        def _reachable(name):
            if (map_arr == ITEMIDX[name]).any():
                return True
            cls = _classes[name]
            for knife in self.knife:
                if knife.holding is not None and isinstance(knife.holding, cls):
                    return True
            for ag in self.agent:
                if ag.holding is not None and isinstance(ag.holding, cls):
                    return True
            return False

        _item_present = {n: _reachable(n) for n in _classes}
        for idx, agent in enumerate(self.agent):
            macro_agent = self.macroAgent[idx]
            if macro_agent.cur_macro_action_done:
                continue
            macro_action = macro_agent.cur_macro_action
            action_name = macroActionName[macro_action]
            is_food_get = action_name == "get tomato" or action_name == "get lettuce" \
                          or action_name == "get onion" or action_name == "get peas"
            is_plate_get = action_name == "get plate 1" or action_name == "get plate 2"

            # Early-termination: if the macro targets an item no longer on the
            # map, end it now so the agent can pick something else next step.
            # Cheaper and policy-compatible alternative to action masking.
            if is_food_get:
                raw = action_name[4:]  # "get tomato" -> "tomato"
                if not _item_present.get(raw, True):
                    macro_agent.cur_macro_action_done = True
                    continue
            elif is_plate_get and not _item_present["plate"]:
                macro_agent.cur_macro_action_done = True
                continue
            if (action_name == "go to knife 1" or action_name == "go to knife 2") and not agent.holding:
                target_x, target_y = self._findPOitem(agent, macro_action)
                if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                    macro_agent.cur_macro_action_done = True
            elif is_food_get:
                target_x, target_y = self._findPOitem(agent, macro_action)
                if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                    # Map action name → rawName without rebuilding a dict each call.
                    raw = action_name[4:]  # drop "get "
                    for knife in self.knife:
                        if knife.x == target_x and knife.y == target_y:
                            food = self._findItem(target_x, target_y, raw)
                            if food is not None and not food.chopped:
                                macro_agent.cur_macro_action_done = True
                                break
            elif action_name == "deliver" and not agent.holding:
                target_x, target_y = self._findPOitem(agent, macro_action)
                if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                    macro_agent.cur_macro_action_done = True
            elif mapType in ("B", "C") and action_name == "go to counter " and not agent.holding:
                target_x = 0
                target_y = int(self.ylen // 2)
                findEmptyCounter = False
                for i in self.counterSequence:
                    if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                        target_x = i
                        findEmptyCounter = True
                        break
                if findEmptyCounter:
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        macro_agent.cur_macro_action_done = True
                else:
                    macro_agent.cur_macro_action_done = True

            if is_food_get or is_plate_get:
                target_x, target_y = self._findPOitem(agent, macro_action)
                if action_name == "get tomato":
                    item = self.tomato[0] if self.tomato else None
                elif action_name == "get lettuce":
                    item = self.lettuce[0] if self.lettuce else None
                elif action_name == "get onion":
                    item = self.onion[0] if self.onion else None
                elif action_name == "get peas":
                    item = self.peas[0] if self.peas else None
                elif action_name == "get plate 1":
                    item = self.plate[0] if self.plate else None
                elif action_name == "get plate 2":
                    item = self.plate[1] if len(self.plate) > 1 else None
                else:
                    item = None
                if item is not None and (target_x != item.x or target_y != item.y):
                    macro_agent.cur_macro_action_done = True

    def _computeLowLevelActions(self, macro_actions):

        """
        Parameters
        ----------
        macro_actions : int | List[..]
            The discrete macro-actions index for the agents. 

        Returns
        -------
        primitive_actions : int | List[..]
            The discrete primitive-actions index for the agents. 
        """

        primitive_actions = []
        stay_idx = self._stay_action_idx
        right_idx = self._right_idx
        macroActionName = self.macroActionName
        mapType = self.mapType

        for idx, agent in enumerate(self.agent):
            macro_agent = self.macroAgent[idx]
            if macro_agent.cur_macro_action_done:
                macro_agent.cur_macro_action = macro_actions[idx]
                macro_action = macro_actions[idx]
                macro_agent.cur_macro_action_done = False
            else:
                macro_action = macro_agent.cur_macro_action

            primitive_action = stay_idx
            # Fetch action name ONCE — the original did ~8 indexed lookups
            # of self.macroActionName[macro_action] per agent per step.
            action_name = macroActionName[macro_action]

            if macro_action == 0:
                macro_agent.cur_macro_action_done = True
            elif action_name == "get plate 2" and len(self.plate) < 2:
                # Map A only has 1 plate, so "get plate 2" becomes a no-op
                macro_agent.cur_macro_action_done = True
            elif action_name == "chop":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]]
                    if new_name == "knife":
                        knife = self._findItem(new_x, new_y, new_name)
                        if knife is not None and isinstance(knife.holding, Food):
                            if not knife.holding.chopped:
                                primitive_action = action
                                macro_agent.cur_chop_times += 1
                                if macro_agent.cur_chop_times >= 3:
                                    macro_agent.cur_macro_action_done = True
                                    macro_agent.cur_chop_times = 0
                                break
                if primitive_action == stay_idx:
                    macro_agent.cur_macro_action_done = True
            elif action_name == "deliver" and agent.x == 1 and agent.y == 1 and ITEMNAME[agent.pomap[2][1]] == "agent":
                primitive_action = ACTIONIDX["right"]
            elif mapType in ("B", "C") and action_name == "go to counter":
                findEmptyCounter = False
                target_x = 0
                target_y = int(self.ylen // 2)
                for i in self.counterSequence:
                    if ITEMNAME[agent.pomap[i][target_y]] == "counter":
                        target_x = i
                        findEmptyCounter = True
                        break
                if findEmptyCounter:
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        macro_agent.cur_macro_action_done = True
                else:
                    primitive_action = stay_idx
                    macro_agent.cur_macro_action_done = True
            elif mapType == "D" and action_name == "go to blender":
                # Navigate to the blender
                if self.blender:
                    target_x = self.blender[0].x
                    target_y = self.blender[0].y
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        if agent.holding and isinstance(agent.holding, Food) and agent.holding.chopped:
                            for action in range(4):
                                if agent.x + DIRECTION[action][0] == target_x and agent.y + DIRECTION[action][1] == target_y:
                                    primitive_action = action
                                    break
                            macro_agent.cur_macro_action_done = True
                        elif agent.holding and isinstance(agent.holding, Peas):
                            for action in range(4):
                                if agent.x + DIRECTION[action][0] == target_x and agent.y + DIRECTION[action][1] == target_y:
                                    primitive_action = action
                                    break
                            macro_agent.cur_macro_action_done = True
                        elif agent.holding and isinstance(agent.holding, BlendedBowl):
                            macro_agent.cur_macro_action_done = True
                        else:
                            blender = self.blender[0]
                            if blender.blended:
                                for action in range(4):
                                    if agent.x + DIRECTION[action][0] == target_x and agent.y + DIRECTION[action][1] == target_y:
                                        primitive_action = action
                                        break
                            macro_agent.cur_macro_action_done = True
                else:
                    macro_agent.cur_macro_action_done = True
            elif mapType == "D" and action_name == "blend":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]]
                    if new_name == "blender":
                        blender = self._findItem(new_x, new_y, new_name)
                        if blender and blender.can_blend() and not blender.blended:
                            primitive_action = action
                            macro_agent.cur_blend_times += 1
                            if macro_agent.cur_blend_times >= 5:
                                macro_agent.cur_macro_action_done = True
                                macro_agent.cur_blend_times = 0
                            break
                if primitive_action == stay_idx:
                    macro_agent.cur_macro_action_done = True
                    macro_agent.cur_blend_times = 0
            elif mapType == "D" and (action_name == "go to oven 1" or action_name == "go to oven 2"):
                oven_idx = 0 if action_name == "go to oven 1" else 1
                if len(self.oven) > oven_idx:
                    target_x = self.oven[oven_idx].x
                    target_y = self.oven[oven_idx].y
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if self._calDistance(agent.x, agent.y, target_x, target_y) == 1:
                        oven = self.oven[oven_idx]
                        if agent.holding and isinstance(agent.holding, BlendedBowl):
                            for action in range(4):
                                if agent.x + DIRECTION[action][0] == target_x and agent.y + DIRECTION[action][1] == target_y:
                                    primitive_action = action
                                    break
                            macro_agent.cur_macro_action_done = True
                        elif agent.holding and isinstance(agent.holding, Plate) and oven.cooked:
                            for action in range(4):
                                if agent.x + DIRECTION[action][0] == target_x and agent.y + DIRECTION[action][1] == target_y:
                                    primitive_action = action
                                    break
                            macro_agent.cur_macro_action_done = True
                        else:
                            macro_agent.cur_macro_action_done = True
                else:
                    macro_agent.cur_macro_action_done = True
            elif mapType == "D" and action_name == "cook":
                for action in range(4):
                    new_x = agent.x + DIRECTION[action][0]
                    new_y = agent.y + DIRECTION[action][1]
                    new_name = ITEMNAME[self.map[new_x][new_y]]
                    if new_name == "oven":
                        oven = self._findItem(new_x, new_y, new_name)
                        if oven and oven.cooking and not oven.cooked:
                            primitive_action = action
                            macro_agent.cur_cook_times += 1
                            if macro_agent.cur_cook_times >= 10:
                                macro_agent.cur_macro_action_done = True
                                macro_agent.cur_cook_times = 0
                            break
                if primitive_action == stay_idx:
                    macro_agent.cur_macro_action_done = True
                    macro_agent.cur_cook_times = 0
            elif macro_action >= right_idx:
                macro_agent.cur_macro_action_done = True
                action = macro_action - right_idx
                new_x = agent.x + DIRECTION[action][0]
                new_y = agent.y + DIRECTION[action][1]
                # Compare int cell directly instead of ITEMNAME[...] == "space"
                if agent.pomap[new_x][new_y] == 0:
                    primitive_action = action
                else:
                    primitive_action = stay_idx
            else:
                target_x, target_y = self._findPOitem(agent, macro_action)

                inPlate = False
                if action_name == "get tomato" or action_name == "get lettuce" or action_name == "get onion" \
                        or action_name == "get peas" or action_name == "get blended bowl" or action_name == "get patty":
                    if (target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) \
                        or self.obs_radius == 0:
                        for plate in self.plate:
                            if plate.x == target_x and plate.y == target_y:
                                primitive_action = stay_idx
                                macro_agent.cur_macro_action_done = True
                                inPlate = True
                                break
                if inPlate:
                    primitive_actions.append(primitive_action)
                    continue

                if target_x == 1 and target_y == 0 and agent.x == 3 and agent.y == 1 and ITEMNAME[agent.pomap[2][1]] == "agent":
                    primitive_action = ACTIONIDX["right"]
                elif ITEMNAME[agent.pomap[target_x][target_y]] == "agent" \
                    and ((target_x >= agent.x - self.obs_radius and target_x <= agent.x + self.obs_radius and target_y >= agent.y - self.obs_radius and target_y <= agent.y + self.obs_radius) or self.obs_radius == 0):
                    macro_agent.cur_macro_action_done = True
                else:
                    primitive_action = self._navigate(agent, target_x, target_y)
                    if primitive_action == stay_idx:
                        macro_agent.cur_macro_action_done = True
                    dist = self._calDistance(agent.x, agent.y, target_x, target_y)
                    if dist == 0:
                        macro_agent.cur_macro_action_done = True
                    elif dist == 1:
                        macro_agent.cur_macro_action_done = True
                        is_plate = action_name == "get plate 1" or action_name == "get plate 2"
                        if is_plate and agent.holding:
                            if isinstance(agent.holding, Food):
                                if agent.holding.chopped:
                                    macro_agent.cur_macro_action_done = False
                                else:
                                    primitive_action = stay_idx

                        if (action_name == "go to knife 1" or action_name == "go to knife 2") and not agent.holding:
                            primitive_action = stay_idx

                        is_food_get = action_name == "get tomato" or action_name == "get lettuce" \
                                      or action_name == "get onion" or action_name == "get peas"
                        if is_food_get:
                            for knife in self.knife:
                                if knife.x == target_x and knife.y == target_y:
                                    if isinstance(knife.holding, Food):
                                        if not knife.holding.chopped:
                                            primitive_action = stay_idx
                                            break

                        if is_food_get or is_plate:
                            # Cheap inline lookup — avoids rebuilding the macroAction2Item
                            # dict on every step in the hot path.
                            if action_name == "get tomato":
                                item = self.tomato[0] if self.tomato else None
                            elif action_name == "get lettuce":
                                item = self.lettuce[0] if self.lettuce else None
                            elif action_name == "get onion":
                                item = self.onion[0] if self.onion else None
                            elif action_name == "get peas":
                                item = self.peas[0] if self.peas else None
                            elif action_name == "get plate 1":
                                item = self.plate[0] if self.plate else None
                            elif action_name == "get plate 2":
                                item = self.plate[1] if len(self.plate) > 1 else None
                            else:
                                item = None
                            if item is not None and (target_x != item.x or target_y != item.y):
                                primitive_action = stay_idx

            primitive_actions.append(primitive_action)
        return primitive_actions
           
    # A star
    def _get_pomap_hash(self, pomap):
        """Cheap hashable representation of a row-major list-of-lists pomap.

        `bytes(iterable)` with small non-negative ints avoids the Python list
        round-trip + tuple allocation of the previous implementation.
        """
        return bytes(itertools.chain.from_iterable(pomap))

    # Class-level constants so the A* inner loop avoids repeated dict lookups.
    _SPACE_IDX = 0   # ITEMIDX["space"]
    _AGENT_IDX = 2   # ITEMIDX["agent"]
    _N_ITEMS = len(ITEMNAME)

    def _navigate(self, agent, target_x, target_y):

        """
        Parameters
        ----------
        agent : Agent
            The current agent.
        target_x : int
            X position of the target item.
        target_y : int
            Y position of the target item.

        Returns
        -------
        action : int
            The primitive-action for the agent to choose.
        """

        # Reuse pomap hash across multiple _navigate calls within the same step.
        # `_get_vector_obs` invalidates agent._pomap_hash each time pomap is rebuilt.
        pomap_hash = getattr(agent, "_pomap_hash", None)
        if pomap_hash is None:
            pomap_hash = self._get_pomap_hash(agent.pomap)
            agent._pomap_hash = pomap_hash
        cache_key = (agent.x, agent.y, target_x, target_y, pomap_hash)
        cached = self._astar_cache.get(cache_key)
        if cached is not None:
            self._astar_cache_hits += 1
            return cached
        self._astar_cache_misses += 1

        direction = [(0,1), (0,-1), (1,0), (-1,0)]
        actionIdx = [0, 2, 1, 3]

        # Nodes packed directly into heap tuples as (f, pass_agent, counter, x, y, g, init_action).
        # This avoids the per-node AStarAgent instantiation (155k+ allocations in a short run).
        counter_it = itertools.count()
        ax0 = agent.x
        ay0 = agent.y
        start_dis = abs(target_x - ax0) + abs(target_y - ay0)
        q = [(start_dis, 0, next(counter_it), ax0, ay0, 0, None)]
        ylen = self.ylen
        isVisited = bytearray(self.xlen * ylen)
        isVisited[ax0 * ylen + ay0] = 1
        pomap = agent.pomap
        space_idx = self._SPACE_IDX
        agent_idx = self._AGENT_IDX
        n_items = self._N_ITEMS
        heappush = heapq.heappush
        heappop = heapq.heappop

        while q:
            _, _, _, aX, aY, aG, prev_action = heappop(q)

            for action in range(4):
                dx, dy = direction[action]
                new_x = aX + dx
                new_y = aY + dy
                visited_idx = new_x * ylen + new_y
                if isVisited[visited_idx]:
                    continue

                pomap_value = pomap[new_x][new_y]
                # Direct int comparisons instead of ITEMNAME[...] lookup + string compare.
                is_agent = (pomap_value == agent_idx)
                passable = is_agent or pomap_value == space_idx or pomap_value >= n_items

                init_action = prev_action if prev_action is not None else actionIdx[action]

                if passable:
                    pass_agent = 1 if is_agent else 0
                    g = aG + 1
                    # Inline Manhattan distance — saves one Python function call per expansion.
                    f = g + abs(target_x - new_x) + abs(target_y - new_y)
                    heappush(q, (f, pass_agent, next(counter_it), new_x, new_y, g, init_action))
                    isVisited[visited_idx] = 1
                if new_x == target_x and new_y == target_y:
                    self._astar_cache[cache_key] = init_action
                    return init_action
        #if no path found, stay
        self._astar_cache[cache_key] = ACTIONIDX["stay"]
        return ACTIONIDX["stay"]

    def _calDistance(self, x, y, target_x, target_y):
        return abs(target_x - x) + abs(target_y - y)
    
    def _calItemDistance(self, agent, item):
        return abs(item.x - agent.x) + abs(item.y - agent.y)

    def _collectCurMacroActions(self):
        # loop each agent
        cur_mac = []
        for agent in self.macroAgent:
            cur_mac.append(agent.cur_macro_action)
        return cur_mac

    def _computeMacroActionDone(self):
        # loop each agent
        mac_done = []
        for agent in self.macroAgent:
            mac_done.append(agent.cur_macro_action_done)
        return mac_done

    def _get_macro_obs(self):

        """
        Returns
        -------
        macro_obs : list
            observation for each agent.
        """
        if self.mode == "vector":
            return self._get_macro_vector_obs()
        elif self.mode == "image":
            return self._get_macro_image_obs()
          

    def _get_macro_vector_obs(self):

        """
        Returns
        -------
        macro_vector_obs : list
            vector observation for each agent.
        """

        macro_obs = []
        obs_radius = self.obs_radius
        inv_x = self._inv_xlen
        inv_y = self._inv_ylen
        item_list = self.itemList
        is_food_tbl = self._is_food
        inv_chop_tbl = self._inv_required_chop
        task_tail = self._task_tail
        task_tail_len = task_tail.size
        obs_size = self._obs_size
        task_head = obs_size - task_tail_len
        n_items = len(item_list)

        for idx, agent in enumerate(self.agent):
            macro_agent = self.macroAgent[idx]
            if macro_agent.cur_macro_action_done:
                obs = np.zeros(obs_size, dtype=np.float32)
                obs_idx = 0
                ax = agent.x
                ay = agent.y
                lo_x = ax - obs_radius
                hi_x = ax + obs_radius
                lo_y = ay - obs_radius
                hi_y = ay + obs_radius
                for i in range(n_items):
                    item = item_list[i]
                    is_food = is_food_tbl[i]
                    ix = item.x
                    iy = item.y
                    if obs_radius == 0 or (lo_x <= ix <= hi_x and lo_y <= iy <= hi_y):
                        obs[obs_idx] = ix * inv_x
                        obs[obs_idx + 1] = iy * inv_y
                        obs_idx += 2
                        if is_food:
                            obs[obs_idx] = item.cur_chopped_times * inv_chop_tbl[i]
                            obs_idx += 1
                    else:
                        obs_idx += 3 if is_food else 2
                obs[task_head:task_head + task_tail_len] = task_tail
                macro_agent.cur_macro_obs = obs
            macro_obs.append(macro_agent.cur_macro_obs.copy())
        return macro_obs

    def _get_macro_image_obs(self):

        """
        Returns
        -------
        macro_image_obs : list
            image observation for each agent.
        """
        
        macro_obs = []
        for idx, agent in enumerate(self.agent):
            if self.macroAgent[idx].cur_macro_action_done:
                if self.game is None:
                    raise RuntimeError("Cannot get image observations when game is not initialized. Use mode='image' or debug=True.")
                frame = self.game.get_image_obs()
                if self.obs_radius > 0:
                    old_image_width, old_image_height, channels = frame.shape

                    new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
                    new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
                    color = (0,0,0)
                    obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

                    x_center = (new_image_width - old_image_width) // 2
                    y_center = (new_image_height - old_image_height) // 2

                    obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame
                    obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)

                    self.macroAgent[idx].cur_macro_obs = obs 
                else:
                    self.macroAgent[idx].cur_macro_obs = frame 
            macro_obs.append(self.macroAgent[idx].cur_macro_obs)
        return macro_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        avail = [1] * self.action_spaces[nth].n
        # Task 9 / Map D: once the patty has been baked, the lettuce / peas /
        # tomato are sealed inside the Patty (or still in the oven turning
        # into one). Selecting "get tomato/lettuce/peas" at that point sends
        # the agent to a stale ingredient position and stalls progress, so
        # mask those macro-actions out.
        if self.task == "lettuce-peas-tomato-patty" and self._patty_baked():
            for action_name in ("get tomato", "get lettuce", "get peas"):
                try:
                    avail[self.macroActionName.index(action_name)] = 0
                except ValueError:
                    pass
        return avail

    def _patty_baked(self):
        for agent in self.agent:
            if isinstance(agent.holding, Patty):
                return True
        for plate in self.plate:
            if plate.containing:
                for item in plate.containing:
                    if isinstance(item, Patty):
                        return True
        for oven in self.oven:
            if oven.cooked:
                return True
        patty_idx = ITEMIDX["patty"]
        for x in range(self.xlen):
            row = self.map[x]
            for y in range(self.ylen):
                if row[y] == patty_idx:
                    return True
        return False