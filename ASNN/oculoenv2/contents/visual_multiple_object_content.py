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

BALL_COLOR = [0.5, 0.5, 0.5]

MAX_STEP_COUNT = 180 * 60

START_STEP_COUNT = 180
MOVING_STEP_COUNT = 180
WAITING_STEP_COUNT = 180

MOVE_REGION_RATE = 0.5

MIN_WIDTH = 0.005
MAX_WIDTH = 1.2

class MultipleObjectTrackingSprite(object):
    def __init__(self, texture, width=0.04, speed=0.04):
        self.tex = texture
        self.width = width
        self.speed = speed

        verts = [-1,-1, 1,-1, -1,1, 1,1]
        texcs = [0,0, 1,0, 0,1, 1,1]
        self.object = pyglet.graphics.vertex_list(4, ('v2f', verts), ('t2f', texcs))

    def circle(self, x, y, radius):
        iterations = 10#int(2*radius*np.pi)
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
        # glFlush()


    def randomize_pos(self, rate_x, rate_y):
        # rate_x = np.random.uniform(low=0.0, high=1.0)
        # rate_y = np.random.uniform(low=0.0, high=1.0)

        self.pos_x = (2.0 * rate_x - 1.0) * MOVE_REGION_RATE
        self.pos_y = (2.0 * rate_y - 1.0) * MOVE_REGION_RATE

    def randomize_direction(self, rate_direction):
        self.direction = rate_direction * np.pi
        # self.direction = np.random.uniform(low=-1.0, high=1.0) * np.pi

    def reflect(self):
        self.direction = np.pi + self.direction

    def render(self):
        glClear(GL_COLOR_BUFFER_BIT)
        glPushMatrix()
        self.circle(self.pos_x, self.pos_y, self.width)
        glPopMatrix()

    def is_conflict_with(self, other_ball_sprites):
        for other_ball_sprite in other_ball_sprites:
            dx = self.pos_x - other_ball_sprite.pos_x
            dy = self.pos_y - other_ball_sprite.pos_y
            dist_sq = dx*dx + dy*dy
            dist_min = self.width*2
            if dist_sq < (dist_min * dist_min):
                return True
        return False

    def _check_illegal(self, ball_index, all_ball_sprites):
        illegal = False

        if self._is_out_of_wall():
            # If hitting the wall
            illegal = True

        for i,other_ball_sprite in enumerate(all_ball_sprites):
            if i == ball_index:
                continue
            dx = self.cand_pos_x - other_ball_sprite.pos_x
            dy = self.cand_pos_y - other_ball_sprite.pos_y
            dist_sq = dx*dx + dy*dy
            dist_min = self.width*2
            if dist_sq < (dist_min * dist_min):
                illegal = True
        return illegal

    def _move_trial(self):
        dx = np.cos(self.direction) * self.speed
        dy = np.sin(self.direction) * self.speed

        # Move to temporal candidate pos
        self.cand_pos_x = self.pos_x + dx
        self.cand_pos_y = self.pos_y + dy

    def move(self, ball_index, all_ball_sprites):
        # Move with current direction
        self._move_trial()
        # Check conflict with other sprites
        conflicted = self._check_illegal(ball_index, all_ball_sprites)
        if conflicted:
            self.reflect()
            # self.randomize_direction()
        # Fix position
        self._fix_pos()
        return 64*np.array([self.pos_x, self.pos_y])+64

    def _is_out_of_wall(self):
        min_pos = -1.0 * 0.9
        max_pos = 1.0 * 0.9

        # Check whether temporal candidate pos is out of wall
        if self.cand_pos_x < min_pos or self.cand_pos_x > max_pos or \
           self.cand_pos_y < min_pos or self.cand_pos_y > max_pos:
            return True
        else:
            return False

    def _fix_pos(self):
        self.pos_x = self.cand_pos_x
        self.pos_y = self.cand_pos_y

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
        # glScalef(self.width, self.width, self.width)
        glScalef(2, 2, 1)
        glRotatef(0.0, 0.0, 0.0, 0.0)
        glBindTexture(self.tex.target, self.tex.id)
        self.bg.draw(GL_TRIANGLE_STRIP)
        # glColorMask(False, False, False, False)
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
    difficulty_range = 6
    def __init__(self):
        super(ObjectMovingBackgroundContent, self).__init__()

        self.difficulty = 2

    def _init(self):
        self.background_texture = self._load_image('black_white_bar.png')
        # self.background_texture = self._load_image('desert.jpg')
        self.target_texture = self._load_texture('general_round0.png')

        self._prepare_background_sprites()
        self._prepare_ball_sprites()

    def _prepare_background_sprites(self):
        # self.background_sprite = BackgroundMovingSprite(self.background_texture, 1.0, 0.5, 0.1)
        self.background_sprite = BackgroundMovingSprite(self.background_texture, 0.0, 1.0, 0.0)

    def _prepare_ball_sprites(self):
        ball_size = 5

        ball_sprites = []

        for _ in range(ball_size):
            ball_sprite = MultipleObjectTrackingSprite(self.target_texture, width=0.08, speed=0.01)
            ball_sprites.append(ball_sprite)
        self.ball_sprites = ball_sprites

        initposx = [0.5492213714661789, 0.201689183841683, 0.7404721651641941, 0.16233104976959445, 0.9464206249492129]
        initposy = [0.21681708603692873, 0.201689183841683, 0.7031936265755183, 0.7836455791832099, 0.20712368649022606]
        initdirection = [0.6304167884885055, 0.8856039605051121, 0.8346481752112759, 0.2772728543691204, 0.1805527758192218]
        for i, ball_sprite in enumerate(self.ball_sprites):
            ball_sprite.randomize_pos(initposx[i], initposy[i])
            ball_sprite.randomize_direction(initdirection[i])

            # if i > 0:
            #     for _ in range(10):
            #         if ball_sprite.is_conflict_with(self.ball_sprites[0:i]):
            #             ball_sprite.randomize_pos()
            #         else:
            #             break

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
            for i, ball_sprite in enumerate(self.ball_sprites):
                # Check conflict with wall and other balls, and then move it.
                pos = ball_sprite.move(i, self.ball_sprites)
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
        glClearColor(0.0, 0.0, 0.0, 1)
        for i, ball_sprite in enumerate(self.ball_sprites):
            ball_sprite.render()
        self.background_sprite.render()
