# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyglet.gl import *

from .base_content import BaseContent, ContentSprite

WHITE_COLOR = np.array([1.0, 1.0, 1.0])

PHASE_START = 0
PHASE_MOVING = 1
PHASE_WAITING = 2

START_MARKER_WIDTH = 0.15

BALL_COLOR = [0.0, 0.0, 0.0]

MAX_STEP_COUNT = 180 * 60

START_STEP_COUNT = 180
MOVING_STEP_COUNT = 180
WAITING_STEP_COUNT = 180

MOVE_REGION_RATE = 0.5

MIN_WIDTH = 0.005
MAX_WIDTH = 1.2

class ObjectMovingSprite(object):
    def __init__(self, texture):
        self.tex = texture
        self.width = 0.08
        verts = [-1,-1, 1,-1, -1,1, 1,1]
        texcs = [0,0, 1,0, 0,1, 1,1]
        self.object = pyglet.graphics.vertex_list(4, ('v2f', verts), ('t2f', texcs))

    def randomize_pos(self):
        rate_x = np.random.uniform(low=0.0, high=1.0)
        rate_y = np.random.uniform(low=0.0, high=1.0)

        self.pos_x = (2.0 * rate_x - 1.0) * MOVE_REGION_RATE
        self.pos_y = (2.0 * rate_y - 1.0) * MOVE_REGION_RATE

    def render(self):
        glColor3f(*BALL_COLOR)
        glPushMatrix()
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glTranslatef(self.pos_x, self.pos_y, 0)
        glScalef(self.width, self.width, self.width)
        glBindTexture(self.tex.target, self.tex.id)
        self.object.draw(GL_TRIANGLE_STRIP)
        glDisable(GL_BLEND)
        glPopMatrix()

    def move(self, t):
        self.pos_x += 0.02
        # self.pos_y += 0.02

class ObjectMovingCircleSprite(object):
    def __init__(self, texture, radius=0.04, speed=0.04, psi=0):
        self.tex = texture
        self.radius = radius
        self.speed = speed
        self.psi = psi
        verts = [-1,-1, 1,-1, -1,1, 1,1]
        texcs = [0,0, 1,0, 0,1, 1,1]
        self.object = pyglet.graphics.vertex_list(4, ('v2f', verts), ('t2f', texcs))

    def circle(self, x, y, radius):
        iterations = 10#int(2*radius*np.pi)
        s = np.sin(2*np.pi / iterations)
        c = np.cos(2*np.pi / iterations)

        dx, dy = radius, 0

        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        for _ in range(iterations+1):
            glVertex2f(x+dx, y+dy)
            dx, dy = (dx*c - dy*s), (dy*c + dx*s)
        glEnd()

    def randomize_pos(self):
        rate_x = np.random.uniform(low=0.0, high=1.0)
        rate_y = np.random.uniform(low=0.0, high=1.0)

        # self.pos_x = (2.0 * rate_x - 1.0) * MOVE_REGION_RATE
        # self.pos_x = (rate_x + 1.0) * MOVE_REGION_RATE
        # self.pos_y = (2.0 * rate_y - 1.0) * MOVE_REGION_RATE
        self.pos_x = 0.8
        self.pos_y = 0

    def render(self):
        glColor3f(*BALL_COLOR)
        glPushMatrix()
        self.circle(self.pos_x, self.pos_y, self.radius)
        glPopMatrix()

    def move(self, t):
        self.pos_x -= self.speed#/np.sqrt(2)
        # self.pos_y -= self.speed/np.sqrt(2)
        self.pos_y += 2*self.speed*np.sin(2*np.pi*t/20 + self.psi)
        return 64*np.array([self.pos_x, self.pos_y])+64

class BackgroundMovingSprite(object):
    def __init__(self, tex, pos_x=0.0, pos_y=0.0, width=1.0, color=[1.0, 1.0, 1.0]):
        self.tex = tex
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.width = width
        self.color = color
        verts = [-1,-1, 1,-1, -1,1, 1,1]
        texcs = [0,0, 1,0, 0,1, 1,1]
        self.bg = pyglet.graphics.vertex_list(4, ('v2f', verts), ('t2f', texcs))

    def render(self):
        glColor3f(*self.color)
        glPushMatrix()
        glTranslatef(self.pos_x, self.pos_y, 0.0)
        # glScalef(self.width, self.width, self.width)
        # glScalef(4.0, 1.0, 1) # forest
        glScalef(3.0, 2.0, 1) # grassland
        glRotatef(0.0, 0.0, 0.0, 0.0)
        glBindTexture(self.tex.target, self.tex.id)
        self.bg.draw(GL_TRIANGLE_STRIP)
        glPopMatrix()

    def set_pos(self, pos):
        self.pos_x = pos[0]
        self.pos_y = pos[1]

    def set_width(self, width):
        self.width = width

    def move(self, t):
        self.pos_x += 0.004
        # self.pos_y += 0.02

class ObjectMovingBackgroundContent(BaseContent):
    def __init__(self):
        super(ObjectMovingBackgroundContent, self).__init__()

    def _init(self):
        self.background_texture = self._load_image('grassland.jpg')
        self.target_texture = self._load_texture('white0.png')

        self._prepare_background_sprites()
        self._prepare_target_sprites()

    def _prepare_background_sprites(self):
        # self.background_sprite = BackgroundMovingSprite(self.background_texture, 1.0, 0.0, 1.0)
        self.background_sprite = BackgroundMovingSprite(self.background_texture, 1.0, 0.5, 1.0)

    def _prepare_target_sprites(self):
        self.target_sprites = []
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=0))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=np.pi/4))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=np.pi/2))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=3*np.pi/4))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=np.pi))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=5*np.pi/4))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=3*np.pi/2))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=7*np.pi/4))
        self.target_sprites.append(ObjectMovingCircleSprite(self.target_texture, radius=0.03, speed=0.03, psi=2*np.pi))
        for i in range(len(self.target_sprites)):
            self.target_sprites[i].randomize_pos()

    def _reset(self):
        self.phase = PHASE_START
        self.phase_count = 0

    def _step(self, local_focus_pos):
        reward = 0

        need_render = False

        info = []
        if self.phase == PHASE_START:
            # if self.start_sprite.contains(local_focus_pos):

            # self._prepare_bar_sprites()
            self.phase_count += 1
            self.phase = PHASE_MOVING
            self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_MOVING:
            # Move balls
            for i in range(len(self.target_sprites)):
                pos = self.target_sprites[i].move(self.phase_count)
                if (pos > 2).all() & (pos < 126).all():
                    info.append(pos)
            self.background_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= MOVING_STEP_COUNT:
                self.phase = PHASE_WAITING
                self.phase_count = 0
            need_render = True
        else:
            self.phase_count += 1
            if self.phase_count >= WAITING_STEP_COUNT:
                self.phase = PHASE_START
                self.phase_count = 0
            need_render = True

        done = self.step_count >= (MAX_STEP_COUNT - 1)
        return reward, done, need_render, info

    def _render(self):
        for i in range(len(self.target_sprites)):
            self.target_sprites[i].render()
        self.background_sprite.render()
