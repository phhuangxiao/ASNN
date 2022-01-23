# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyglet.gl import *

from .base_content import BaseContent, ContentSprite

PHASE_START = 0
PHASE_LOOMING = 1
PHASE_WAITING = 2
PHASE_AWAY = 3
PHASE_RESPONSE = 4

START_MARKER_WIDTH = 0.15

BALL_COLOR = [0.0, 0.0, 0.0]

MAX_STEP_COUNT = 180 * 60

START_STEP_COUNT = 10
LOOMING_STEP_COUNT = 45
WAITING_STEP_COUNT = 20
AWAY_STEP_COUNT = 45

MOVE_REGION_RATE = 0.0

MIN_WIDTH = 0.005
MAX_WIDTH = 1.2


class ObjectLoomingSprite(object):
    def __init__(self, texture):
        self.tex = texture
        self.width = 0.01 #0.05

        verts = [-1,-1, 1,-1, -1,1, 1,1]
        texcs = [0,0, 1,0, 0,1, 1,1]
        self.object = pyglet.graphics.vertex_list(4, ('v2f', verts), ('t2f', texcs))

    def randomize_pos(self):
        rate_x = np.random.uniform(low=0.0, high=1.0)
        rate_y = np.random.uniform(low=0.0, high=1.0)

        self.pos_x = (2.0 * rate_x - 1.0) * MOVE_REGION_RATE
        self.pos_y = (2.0 * rate_y - 1.0) * MOVE_REGION_RATE

        # self.pos_x = -0.8
        # self.pos_y = -0.8
    
    def circle(self, x, y, radius):
        iterations = 100
        s = np.sin(2*np.pi / iterations)
        c = np.cos(2*np.pi / iterations)

        dx, dy = radius, 0
        glBegin(GL_TRIANGLE_FAN)
        # glAlphaFunc(GL_GREATER, 0.5)
        glColor3f(*BALL_COLOR)
        glVertex2f(x, y)
        for _ in range(iterations+1):
            glVertex2f(x+dx, y+dy)
            dx, dy = (dx*c - dy*s), (dy*c + dx*s)
        glEnd()

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glPushMatrix()
        self.circle(self.pos_x, self.pos_y, self.width)
        glPopMatrix()

    def loom(self, t):
        self.width += 0.02*np.exp(t/1000)
        # self.width = min(self.width, MAX_WIDTH)
        # self.pos_x += 0.03*np.cos(np.pi/4)
        # self.pos_y += 0.03*np.sin(np.pi/4)

    def away(self, t):
        self.width -= 0.02*np.exp(t/1000)
        # self.width = max(self.width, MIN_WIDTH)
        # self.pos_x -= 0.03*np.cos(np.pi/4)
        # self.pos_y -= 0.03*np.sin(np.pi/4)

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
        # glColor3f(*self.color)
        # glClearColor(1, 1, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT)
        glColor3f(*self.color)
        glPushMatrix()
        glTranslatef(self.pos_x, self.pos_y, 0.0)
        glScalef(3.0, 2.0, 1) 
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
        self.pos_x += 0.01

class ObjectLoomingBackgroundContent(BaseContent):
    def __init__(self):
        super(ObjectLoomingBackgroundContent, self).__init__()

    def _init(self):
        self.ball_texture = self._load_texture('general_round0.png')
        self.background_texture = self._load_image('blackandwhite.jpg')

        self._prepare_background_sprites()
        self._prepare_ball_sprites()
    
    def _prepare_background_sprites(self):
        self.background_sprite = BackgroundMovingSprite(self.background_texture, 0.0, 1.0, 1.0)

    def _prepare_ball_sprites(self):
        self.ball_sprite = ObjectLoomingSprite(self.ball_texture)
        self.ball_sprite.randomize_pos()

    def _reset(self):
        self.phase = PHASE_START
        self.phase_count = 0

    def _step(self, local_focus_pos):
        reward = 0

        need_render = False

        info = {}
        if self.phase == PHASE_START:
            self._prepare_ball_sprites()
            self.phase_count += 1
            if self.phase_count >= START_STEP_COUNT:
                self.phase = PHASE_LOOMING
                self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_LOOMING:
            # Move balls
            self.ball_sprite.loom(self.phase_count)
            self.background_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= LOOMING_STEP_COUNT:
                self.phase = PHASE_WAITING
                self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_WAITING:
            self.background_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= WAITING_STEP_COUNT:
                self.phase = PHASE_AWAY
                self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_AWAY:
            # Move balls
            self.ball_sprite.away(self.phase_count)
            self.background_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= AWAY_STEP_COUNT:
                self.phase = PHASE_RESPONSE
                self.phase_count = 0
            need_render = True
        else:
            self.background_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= WAITING_STEP_COUNT:
                self.phase = PHASE_START
                self.phase_count = 0

        done = self.step_count >= (MAX_STEP_COUNT - 1)
        return reward, done, need_render, info

    def _render(self):
        glClearColor(0.0, 0.0, 0.0, 1)
        self.ball_sprite.render()
        self.background_sprite.render()