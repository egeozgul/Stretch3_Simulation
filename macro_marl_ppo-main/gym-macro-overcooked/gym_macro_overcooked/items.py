#!/usr/bin/python

import numpy as np

class Item(object):
    def __init__(self, pos_x, pos_y):
        self.x = pos_x
        self.y = pos_y

class MovableItem(Item):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.initial_x = pos_x
        self.initial_y = pos_y
    
    def move(self, x, y):
        self.x = x
        self.y = y
    
    def refresh(self):
        self.x = self.initial_x
        self.y = self.initial_y

class Food(MovableItem):
    # 0 for unchoopped 1 for chopped
    def __init__(self, pos_x, pos_y, chopped = False):
        super().__init__(pos_x, pos_y)
        self.chopped = chopped
        self.cur_chopped_times = 0
        self.required_chopped_times = 3
        self.rawName = ""
        # One-shot subtask credit flags. Survive refresh() (which fires on
        # wrong delivery and would otherwise let the agent re-earn the chop
        # and plate rewards by recycling the same item). _createItems()
        # rebuilds Food objects on env.reset(), so these clear per episode.
        self._chop_rewarded = False
        self._plate_rewarded = False

    def chop(self):
        if not self.chopped:
            self.cur_chopped_times += 1
            if self.cur_chopped_times >= self.required_chopped_times:
                self.chopped = True

    def refresh(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.chopped = False
        self.cur_chopped_times = 0

class Tomato(Food):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.rawName = "tomato"
    
    @property
    def name(self):
        if self.chopped:
            return "ChoppedTomato"
        else:
            return "FreshTomato"

class Peas(Food):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.rawName = "peas"
    
    @property
    def name(self):
        if self.chopped:
            return "ChoppedPeas"
        else:
            return "FreshPeas"
            
class Lettuce(Food):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.rawName = "lettuce"
    
    @property
    def name(self):
        if self.chopped:
            return "ChoppedLettuce"
        else:
            return "FreshLettuce"

class Onion(Food):
    def __init__(self, pos_x, pos_y):
        super().__init__(pos_x, pos_y)
        self.rawName = "onion"
    
    @property
    def name(self):
        if self.chopped:
            return "ChoppedOnion"
        else:
            return "FreshOnion"

class FixedItem(Item):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y)
        self.holding = holding

    def hold(self, items):
        self.holding = items
    
    def release(self):
        self.holding = None

class Knife(FixedItem):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y, holding)
        self.rawName = "knife"
    
    @property
    def name(self):
        return "cutboard"

class Delivery(FixedItem):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y, holding)
        self.rawName = "delivery"

    @property
    def name(self):
        return "delivery"

class BlendedBowl(Food):
    """A bowl containing blended ingredients that can be picked up and put in oven"""
    def __init__(self, pos_x, pos_y, containing=None):
        super().__init__(pos_x, pos_y, chopped=True)  # BlendedBowl is considered "processed" like chopped food
        self.containing = containing if containing else []
        self.rawName = "blended_bowl"

    def chop(self):
        """BlendedBowl cannot be chopped — it is a finished item."""
        pass
    
    @property
    def name(self):
        return "blenderBlended"

class Patty(Food):
    """A cooked patty that can be picked up and delivered (behaves like BlendedBowl)"""
    def __init__(self, pos_x, pos_y, containing=None):
        super().__init__(pos_x, pos_y, chopped=True)  # Patty is considered "processed" like chopped food
        self.containing = containing if containing else []  # The ingredients in the patty
        self.rawName = "patty"

    def chop(self):
        """Patty cannot be chopped — it is a finished item."""
        pass

    @property
    def name(self):
        return "lettuceTomatoPeapatty"


class Blender(FixedItem):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y, holding)
        self.rawName = "blender"
        self.containing = []  # List of foods in the blender
        self.blended = False  # Whether the contents have been blended
        self.cur_blend_times = 0
        self.required_blend_times = 5  # Takes 5 timesteps to blend
        # One-shot subtask credit. Wrong-delivery refreshes ingredients and
        # release()/create_blended_bowl() reset the blender, so an ungated
        # reward could be re-earned per cycle. The flag persists across those
        # resets and is only cleared by env.reset() (which rebuilds Blender).
        self._blend_rewarded = False
    
    def can_add_food(self, food):
        """Check if food can be added to blender (lettuce/tomato must be chopped)"""
        if isinstance(food, (Patty, BlendedBowl)):
            return False
        if isinstance(food, (Tomato, Lettuce)):
            return food.chopped
        return True  # Peas and other foods don't need to be chopped
    
    def add_food(self, food):
        """Add a food item to the blender (only if it meets requirements)"""
        if self.can_add_food(food):
            self.containing.append(food)
            food.move(self.x, self.y)
            return True
        return False
    
    def can_blend(self):
        """Check if blender has the required ingredients (all must be chopped)"""
        has_peas = any(isinstance(f, Peas) for f in self.containing)
        has_tomato = any(isinstance(f, Tomato) and f.chopped for f in self.containing)
        has_lettuce = any(isinstance(f, Lettuce) and f.chopped for f in self.containing)
        return has_peas and has_tomato and has_lettuce
    
    def blend_step(self):
        """Perform one step of blending. Returns True if blending is complete."""
        if not self.can_blend():
            return False
        if self.blended:
            return True  # Already blended
        self.cur_blend_times += 1
        if self.cur_blend_times >= self.required_blend_times:
            self.blended = True
            return True
        return False
    
    def create_blended_bowl(self):
        """Create a BlendedBowl from the blended contents"""
        if self.blended:
            bowl = BlendedBowl(self.x, self.y, self.containing)
            self.containing = []
            self.blended = False
            self.cur_blend_times = 0
            return bowl
        return None
    
    def release(self):
        """Release contents and reset blender"""
        self.containing = []
        self.blended = False
        self.cur_blend_times = 0
        self.holding = None
    
    @property
    def name(self):
        if self.blended:
            return "blenderBlended"
        else:
            return "blenderEmpty"

class Oven(FixedItem):
    def __init__(self, pos_x, pos_y, holding = None):
        super().__init__(pos_x, pos_y, holding)
        self.rawName = "oven"
        self.containing = []  # Contents being cooked
        self.cooking = False  # Whether cooking is in progress
        self.cooked = False   # Whether cooking is complete
        self.cur_cook_times = 0
        self.required_cook_times = 10  # Takes 10 timesteps to cook
        # One-shot subtask credit per oven instance. create_patty() resets
        # the cooking state, so an ungated reward could be re-earned by
        # re-cooking. The flag persists across that reset.
        self._cook_rewarded = False
    
    def add_blended(self, blended_contents):
        """Add blended contents from blender to oven"""
        self.containing = blended_contents
        self.cooking = True
        for food in self.containing:
            food.move(self.x, self.y)
    
    def add_blended_bowl(self, blended_bowl):
        """Add a BlendedBowl to the oven"""
        if not isinstance(blended_bowl, BlendedBowl):
            return False
        self.containing = blended_bowl.containing
        self.cooking = True
        for food in self.containing:
            food.move(self.x, self.y)
        return True
    
    def cook_step(self):
        """Perform one step of cooking. Returns True if cooking is complete."""
        if not self.cooking or self.cooked:
            return self.cooked
        self.cur_cook_times += 1
        if self.cur_cook_times >= self.required_cook_times:
            self.cooked = True
            return True
        return False
    
    def create_patty(self):
        """Create a Patty from the cooked contents"""
        if self.cooked:
            patty = Patty(self.x, self.y, self.containing)
            self.containing = []
            self.cooking = False
            self.cooked = False
            self.cur_cook_times = 0
            return patty
        return None
    
    def release(self):
        """Release contents and reset oven"""
        self.containing = []
        self.cooking = False
        self.cooked = False
        self.cur_cook_times = 0
        self.holding = None
    
    @property
    def name(self):
        if self.cooked:
            return "lettuceTomatoPeapatty"
        elif self.cooking:
            return "oven"  # Still cooking
        else:
            return "oven"


class Plate(MovableItem):
    def __init__(self, pos_x, pos_y, containing = None):
        super().__init__(pos_x, pos_y)
        self.containing = containing
        self.rawName = "plate"
        self.is_patty = False  # Flag to indicate if plate contains a cooked patty from oven
    
    def contain(self, items):
        if self.containing:
            self.containing.append(items)
        else:
            self.containing = [items]
        for item in self.containing:
            item.move(self.x, self.y)
    
    def contain_patty(self, foods):
        """Add cooked patty (from oven) to the plate"""
        self.containing = foods if isinstance(foods, list) else [foods]
        self.is_patty = True
        for item in self.containing:
            item.move(self.x, self.y)
    
    def move(self, x, y):
        super().move(x, y)
        if self.containing:
            for item in self.containing:
                item.move(x, y)

    def release(self):
        self.containing = None
        self.is_patty = False

    def refresh(self):
        super().refresh()
        self.is_patty = False

    @property
    def name(self):
        return "plate"

    @property
    def containedName(self):
        # If this is a cooked patty from the oven, return the patty image name
        if self.is_patty:
            return "lettuceTomatoPeapatty"
        
        # Check if plate contains a Patty or BlendedBowl
        if self.containing:
            for item in self.containing:
                if isinstance(item, Patty):
                    return "lettuceTomatoPeapatty"
                if isinstance(item, BlendedBowl):
                    return "blenderBlended"
        
        dishName = ""
        foodList = [Lettuce, Onion, Peas, Tomato]
        foodInPlate = [-1] * len(foodList)
        
        if self.containing is None:
            return ""
        for f in range(len(self.containing)):
            for i in range(len(foodList)):
                if isinstance(self.containing[f], foodList[i]):
                    foodInPlate[i] = f
        for i in range(len(foodList)):
            if foodInPlate[i] > -1:
                dishName += self.containing[foodInPlate[i]].name + "-"
        return dishName[:-1] if dishName else ""


class Agent(MovableItem):
    def __init__(self, pos_x, pos_y, holding = None, color = None):
        super().__init__(pos_x, pos_y)
        self.holding = holding
        self.color = color
        self.moved = False
        self.obs = None
        self.pomap = None
        self.rawName = "agent"

    def pickup(self,item):
        self.holding = item
        item.move(self.x, self.y)
    
    def putdown(self, x, y):
        if self.holding is not None:
            self.holding.move(x, y)
        self.holding = None

    def move(self, x, y):
        super().move(x, y)
        self.moved = True
        if self.holding:
            self.holding.move(x, y)


