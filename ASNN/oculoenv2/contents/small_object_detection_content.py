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

BALL_COLOR = [0, 0, 0]

MAX_STEP_COUNT = 180 * 60

START_STEP_COUNT = 1
LOOMING_STEP_COUNT = 180
WAITING_STEP_COUNT = 180
AWAY_STEP_COUNT = 180

MOVE_REGION_RATE = 0.5

MIN_WIDTH = 0.005
MAX_WIDTH = 1.2


class SmallObjectDetectionSprite(object):
    def __init__(self, texture):
        self.tex = texture
        self.width = 0.04

    def randomize_pos(self):
        rate_x = np.random.uniform(low=0.0, high=1.0)
        rate_y = np.random.uniform(low=0.0, high=1.0)

        # self.pos_x = (2.0 * rate_x - 1.0) * MOVE_REGION_RATE
        # self.pos_y = (2.0 * rate_y - 1.0) * MOVE_REGION_RATE

        self.pos_x = - MOVE_REGION_RATE
        self.pos_y = - MOVE_REGION_RATE

        self.pos_x = 0.8
        self.pos_y = -0.8

    def render(self, common_quad_vlist):
        glColor3f(*BALL_COLOR)
        glPushMatrix()
        glTranslatef(self.pos_x, self.pos_y, 0)
        glScalef(self.width, self.width, self.width)
        glBindTexture(self.tex.target, self.tex.id)
        common_quad_vlist.draw(GL_QUADS)
        glPopMatrix()

    def move(self, t):
        self.pos_x += 0.08*np.cos(3*np.pi/4)
        self.pos_y += 0.08*np.sin(3*np.pi/4)
        # self.pos_x += 0.03*np.cos(t*np.pi/90)
        # self.pos_y += 0.03*np.sin(t*np.pi/90)
        # self.pos_x += (0.2*t/90)*np.cos(7*np.pi/4)
        # self.pos_y += (0.2*t/90)*np.sin(7*np.pi/4)


class SmallObjectDetectionContent(BaseContent):
    def __init__(self):
        super(SmallObjectDetectionContent, self).__init__()

    def _init(self):
        start_marker_texture = self._load_texture('start_marker0.png')
        white_texture = self._load_texture('white0.png')
        self.ball_texture = self._load_texture('general_round0.png')

        self.start_sprite = ContentSprite(start_marker_texture, 0.0, 0.0, START_MARKER_WIDTH)

    def _prepare_ball_sprites(self):
        self.ball_sprite = SmallObjectDetectionSprite(self.ball_texture)
        self.ball_sprite.randomize_pos()

    def _reset(self):
        self.phase = PHASE_START
        self.phase_count = 0

    def _step(self, local_focus_pos):
        reward = 0

        need_render = False

        info = {}

        if self.phase == PHASE_START:
            if self.start_sprite.contains(local_focus_pos):
                self._prepare_ball_sprites()
                self.phase_count += 1
                if self.phase_count >= START_STEP_COUNT:
                    self.phase = PHASE_LOOMING
                # self.phase_count = 0
                need_render = True
        elif self.phase == PHASE_LOOMING:
            # Move balls
            self.ball_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= LOOMING_STEP_COUNT:
                self.phase = PHASE_WAITING
                self.phase_count = 0
            need_render = True
        elif self.phase == PHASE_WAITING:
            self.phase_count += 1
            if self.phase_count >= WAITING_STEP_COUNT:
                self.phase = PHASE_AWAY
                self.phase_count = 0
        elif self.phase == PHASE_AWAY:
            # Move balls
            self.ball_sprite.move(self.phase_count)
            self.phase_count += 1
            if self.phase_count >= AWAY_STEP_COUNT:
                self.phase = PHASE_RESPONSE
                self.phase_count = 0
            need_render = True
        else:
            self.phase_count += 1
            if self.phase_count >= WAITING_STEP_COUNT:
                self.phase = PHASE_START
                self.phase_count = 0

        done = self.step_count >= (MAX_STEP_COUNT - 1)
        return reward, done, need_render, info

    def _render(self):
        if self.phase == PHASE_START:
            self.start_sprite.render(self.common_quad_vlist)
        else:
            self.ball_sprite.render(self.common_quad_vlist)
        # if self.phase_count == 0:
        #     self.start_sprite.render(self.common_quad_vlist)
        # else:
        #     self.ball_sprite.render(self.common_quad_vlist)
