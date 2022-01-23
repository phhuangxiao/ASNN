# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyglet.gl import *

from .base_content import BaseContent, ContentSprite

PHASE_START = 0
PHASE_MOVING_RIGHT = 1
PHASE_WAITING = 2
PHASE_MOVING_LEFT = 3

START_MARKER_WIDTH = 0.15

BALL_COLOR = [0.0, 0.0, 0.0]

MAX_STEP_COUNT = 180 * 60

START_STEP_COUNT = 10
MOVING_RIGHT_COUNT = 45
WAITING_STEP_COUNT = 20
MOVING_LEFT_COUNT = 45


MOVE_REGION_RATE = 0.5

MIN_WIDTH = 0.005
MAX_WIDTH = 1.2


class ObjectMovingSprite(object):
    def __init__(self, texture):
        self.tex = texture
        self.width = 0.2

    def randomize_pos(self):
        rate_x = np.random.uniform(low=0.0, high=1.0)
        rate_y = np.random.uniform(low=0.0, high=1.0)

        # self.pos_x = (2.0 * rate_x - 1.0) * MOVE_REGION_RATE
        # self.pos_y = (2.0 * rate_y - 1.0) * MOVE_REGION_RATE
        self.pos_x = -0.85
        self.pos_y = 0

    def render(self, common_quad_vlist):
        glColor3f(*BALL_COLOR)
        glPushMatrix()
        glTranslatef(self.pos_x, self.pos_y, 0)
        glScalef(self.width, self.width, self.width)
        glBindTexture(self.tex.target, self.tex.id)
        common_quad_vlist.draw(GL_QUADS)
        glPopMatrix()

    def move_right(self, t):
        self.pos_x += 0.04
        # self.pos_y += 0.02

    def move_left(self, t):
        self.pos_x -= 0.04
        # self.pos_y += 0.02

class ObjectMovingContent(BaseContent):
    def __init__(self):
        super(ObjectMovingContent, self).__init__()

    def _init(self):
        start_marker_texture = self._load_texture('start_marker0.png')
        white_texture = self._load_texture('white0.png')
        # self.bar_texture = self._load_texture('general_round0.png')
        self.bar_texture = self._load_texture('general_v_bar0.png')

        self.start_sprite = ContentSprite(start_marker_texture, 0.0, 0.0, START_MARKER_WIDTH)

        self._prepare_bar_sprites()

    def _prepare_bar_sprites(self):
        self.bar_sprite = ObjectMovingSprite(self.bar_texture)
        self.bar_sprite.randomize_pos()

    def _reset(self):
        self.phase = PHASE_START
        self.phase_count = 0

    def _step(self, local_focus_pos):
        reward = 0

        need_render = False

        info = {}

        if self.phase == PHASE_START:
            self._prepare_bar_sprites()
            self.phase_count += 1
            if self.phase_count >= START_STEP_COUNT:
                self.phase = PHASE_MOVING_RIGHT
                self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_MOVING_RIGHT:
            # Move balls
            self.bar_sprite.move_right(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= MOVING_RIGHT_COUNT:
                self.phase = PHASE_WAITING
                self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_WAITING:
            self.phase_count += 1
            if self.phase_count >= WAITING_STEP_COUNT:
                self.phase = PHASE_MOVING_LEFT
                self.phase_count = 0
            need_render = True
        else:
            self.bar_sprite.move_left(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= MOVING_LEFT_COUNT:
                self.phase = PHASE_START
                self.phase_count = 0
            need_render = True
        done = self.step_count >= (MAX_STEP_COUNT - 1)
        return reward, done, need_render, info

    def _render(self):
        # if self.phase == PHASE_START:
        #     self.start_sprite.render(self.common_quad_vlist)
        # else:
        self.bar_sprite.render(self.common_quad_vlist)
