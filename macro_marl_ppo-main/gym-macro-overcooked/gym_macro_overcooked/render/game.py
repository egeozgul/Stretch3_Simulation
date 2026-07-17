import os
import sys

import pygame
import pygame.surfarray
import numpy as np
from .utils import *
from ..items import Tomato, Lettuce, Peas, Plate, Knife, Delivery, Agent, Food, Blender, Oven, BlendedBowl

graphics_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'graphics'))
_image_library = {}

ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion", "peas", "blender", "oven", "blended_bowl"]
ITEMIDX= {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8, "peas": 9, "blender": 10, "oven": 11, "blended_bowl": 12}

def get_image(path):
    global _image_library
    image = _image_library.get(path)
    if image == None:
        canonicalized_path = path.replace('/', os.sep).replace('\\', os.sep)
        image = pygame.image.load(canonicalized_path)
        _image_library[path] = image
    return image


class Game:
    def __init__(self, env):
        self._running = True
        self.env = env

        # Visual parameters
        self.scale = 80   # num pixels per tile
        self.holding_scale = 0.5
        self.container_scale = 0.7
        self.width = self.scale * self.env.xlen
        self.height = self.scale * self.env.ylen
        self.tile_size = (self.scale, self.scale)
        self.holding_size = tuple((self.holding_scale * np.asarray(self.tile_size)).astype(int))
        self.container_size = tuple((self.container_scale * np.asarray(self.tile_size)).astype(int))
        self.holding_container_size = tuple((self.container_scale * np.asarray(self.holding_size)).astype(int))

        # Lazy initialization - only init pygame when actually needed
        self._pygame_initialized = False
        self.screen = None

    def _ensure_pygame_initialized(self):
        """Initialize pygame only when needed for rendering"""
        if not self._pygame_initialized:
            pygame.init()
            self._pygame_initialized = True

    def on_init(self):
        self._ensure_pygame_initialized()
        if self.play:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            # Create a hidden surface
            self.screen = pygame.Surface((self.width, self.height))
        self._running = True


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_render(self):
        self._ensure_pygame_initialized()
        if pygame.display.get_surface() is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.display.get_surface()
        self.screen.fill(Color.FLOOR)
        for x in range(self.env.xlen):
            for y in range(self.env.ylen):
                sl = self.scaled_location((y, x))
                if self.env.map[x][y] == ITEMIDX["counter"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                elif self.env.map[x][y] == ITEMIDX["delivery"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)                    
                    pygame.draw.rect(self.screen, Color.DELIVERY, fill)
                    self.draw('delivery', self.tile_size, sl)
                    for k in self.env.delivery:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["knife"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)    
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('cutboard', self.tile_size, sl)
                    for k in self.env.knife:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["tomato"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for t in self.env.tomato:
                        if t.x == x and t.y == y:
                            self.draw(t.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["lettuce"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.lettuce:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["onion"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.onion:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["peas"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for p in self.env.peas:
                        if p.x == x and p.y == y:
                            self.draw(p.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["plate"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('plate', self.tile_size, sl)
                    for p in self.env.plate:
                        if p.x == x and p.y == y:
                            if p.containing:
                                self.draw(p.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["blender"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for b in self.env.blender:
                        if b.x == x and b.y == y:
                            self.draw(b.name, self.tile_size, sl)  # Uses blenderEmpty or blenderBlended
                            if b.holding:
                                self.draw(b.holding.name, self.tile_size, sl)
                                if b.holding.name == "plate":
                                    if b.holding.containing:
                                        self.draw(b.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["oven"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for o in self.env.oven:
                        if o.x == x and o.y == y:
                            self.draw(o.name, self.tile_size, sl)  # Uses oven or lettuceTomatoPeapatty
                            if o.holding:
                                self.draw(o.holding.name, self.tile_size, sl)
                                if o.holding.name == "plate":
                                    if o.holding.containing:
                                        self.draw(o.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["agent"]:
                    for agent in self.env.agent:
                        if agent.x == x and agent.y == y:
                            self.draw('agent-{}'.format(agent.color), self.tile_size, sl)
                            if agent.holding:
                                if isinstance(agent.holding, Plate):
                                    self.draw('plate', self.holding_size, self.holding_location((y, x)))
                                    if agent.holding.containing:
                                        self.draw(agent.holding.containedName, self.holding_container_size, self.holding_container_location((y, x)))
                                else:
                                    self.draw(agent.holding.name, self.holding_size, self.holding_location((y, x)))
        
        # Return the screen surface so it can be composed with other UI elements
        # Don't flip here - let the caller decide when to update the display
        return self.screen

    def draw(self, path, size, location):
        if not path:
            print(f"WARN render: empty sprite name at location={location}")
            return
        # Try .png first, then .jpg
        image_path = '{}/{}.png'.format(graphics_dir, path)
        if not os.path.exists(image_path):
            image_path = '{}/{}.jpg'.format(graphics_dir, path)
        
        # Fallback for missing combination images
        if not os.path.exists(image_path) and path == "ChoppedPeas-ChoppedTomato":
            image_path = '{}/{}.png'.format(graphics_dir, "ChoppedPeas")
        
        if not os.path.exists(image_path):
            print(f"WARN render: missing sprite '{path}' -> {image_path}")
            return

        image = pygame.transform.scale(get_image(image_path), size)
        self.screen.blit(image, location)

    def scaled_location(self, loc):
        """Return top-left corner of scaled location given coordinates loc, e.g. (3, 4)"""
        return tuple(self.scale * np.asarray(loc))

    def holding_location(self, loc):
        """Return top-left corner of location where agent holding will be drawn (bottom right corner) given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.holding_scale)).astype(int))

    def container_location(self, loc):
        """Return top-left corner of location where contained (i.e. plated) object will be drawn, given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        return tuple((np.asarray(scaled_loc) + self.scale*(1-self.container_scale)/2).astype(int))

    def holding_container_location(self, loc):
        """Return top-left corner of location where contained, held object will be drawn given coordinates loc, e.g. (3, 4)"""
        scaled_loc = self.scaled_location(loc)
        factor = (1-self.holding_scale) + (1-self.container_scale)/2*self.holding_scale
        return tuple((np.asarray(scaled_loc) + self.scale*factor).astype(int))

    def on_cleanup(self):
        pygame.display.quit()
        pygame.quit()

    def get_image_obs(self):
        self._ensure_pygame_initialized()
        self.screen = pygame.Surface((self.width, self.height))
        self.screen.fill(Color.FLOOR)
        for x in range(self.env.xlen):
            for y in range(self.env.ylen):
                sl = self.scaled_location((y, x))
                if self.env.map[x][y] == ITEMIDX["counter"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                elif self.env.map[x][y] == ITEMIDX["delivery"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)                    
                    pygame.draw.rect(self.screen, Color.DELIVERY, fill)
                    self.draw('delivery', self.tile_size, sl)
                    for k in self.env.delivery:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["knife"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)    
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('cutboard', self.tile_size, sl)
                    for k in self.env.knife:
                        if k.x == x and k.y == y:
                            if k.holding:
                                self.draw(k.holding.name, self.tile_size, sl)
                                if k.holding.name == "plate":
                                    if k.holding.containing:
                                        self.draw(k.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["tomato"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for t in self.env.tomato:
                        if t.x == x and t.y == y:
                            self.draw(t.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["lettuce"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.lettuce:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["onion"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for l in self.env.onion:
                        if l.x == x and l.y == y:
                            self.draw(l.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["peas"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for p in self.env.peas:
                        if p.x == x and p.y == y:
                            self.draw(p.name, self.tile_size, sl)
                elif self.env.map[x][y] == ITEMIDX["plate"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    self.draw('plate', self.tile_size, sl)
                    for p in self.env.plate:
                        if p.x == x and p.y == y:
                            if p.containing:
                                self.draw(p.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["blender"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for b in self.env.blender:
                        if b.x == x and b.y == y:
                            self.draw(b.name, self.tile_size, sl)  # Uses blenderEmpty or blenderBlended
                            if b.holding:
                                self.draw(b.holding.name, self.tile_size, sl)
                                if b.holding.name == "plate":
                                    if b.holding.containing:
                                        self.draw(b.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["oven"]:
                    fill = pygame.Rect(sl[0], sl[1], self.scale, self.scale)
                    pygame.draw.rect(self.screen, Color.COUNTER, fill)
                    pygame.draw.rect(self.screen, Color.COUNTER_BORDER, fill, 1)
                    for o in self.env.oven:
                        if o.x == x and o.y == y:
                            self.draw(o.name, self.tile_size, sl)  # Uses oven or lettuceTomatoPeapatty
                            if o.holding:
                                self.draw(o.holding.name, self.tile_size, sl)
                                if o.holding.name == "plate":
                                    if o.holding.containing:
                                        self.draw(o.holding.containedName, self.container_size, self.container_location((y, x)))
                elif self.env.map[x][y] == ITEMIDX["agent"]:
                    for agent in self.env.agent:
                        if agent.x == x and agent.y == y:
                            self.draw('agent-{}'.format(agent.color), self.tile_size, sl)
                            if agent.holding:
                                if isinstance(agent.holding, Plate):
                                    self.draw('plate', self.holding_size, self.holding_location((y, x)))
                                    if agent.holding.containing:
                                        self.draw(agent.holding.containedName, self.holding_container_size, self.holding_container_location((y, x)))
                                else:
                                    self.draw(agent.holding.name, self.holding_size, self.holding_location((y, x)))

        # Use pygame.surfarray for much faster pixel access (1000x+ faster than manual loops)
        img_rgb = pygame.surfarray.array3d(self.screen)
        # Convert from (width, height, 3) to (height, width, 3) and ensure correct channel order
        img_rgb = np.transpose(img_rgb, (1, 0, 2))
        return img_rgb