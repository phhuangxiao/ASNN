'''
A Bioinspired Approach-Sensitive Neural Network
'''
import argparse
import pickle
import os
import time
import cv2
import math
import numpy as np
from skimage import io
from scipy import signal
from sklearn.cluster import DBSCAN
import multiprocessing as mp
from collections import deque

from oculoenv2.environment import Environment
from oculoenv2.contents.visual_looming_content import ObjectLoomingContent
from oculoenv2.contents.visual_moving_content import ObjectMovingContent
from oculoenv2.contents.small_object_detection_content import SmallObjectDetectionContent
from oculoenv2.contents.visual_multiple_object_content import ObjectMovingBackgroundContent
from oculoenv2.contents.visual_looming_background_content import ObjectLoomingBackgroundContent

import matplotlib.pyplot as plt
import matplotlib.cm as cm
cmap = cm.jet # matlab's style of colormap
cmap.set_bad('k',1.) # choose the color and alpha of the background: (k, w, b, g, r)

GAUSSIAN_KERNEL_SIZE = (3,3)

def sigmoid(x, scale, threshold):
    y = 1. / (1 + np.exp(-scale * (x - threshold)))
    return y

def relu(x):
    y = 0.5 * (abs(x) + x)
    return y

def negrelu(x):
    y = 0.5 * (abs(x) - x)
    return y

class Retina(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Bipolar cells
        self.Bipolar_ON = np.zeros((6, self.height, self.width))
        self.Bipolar_OFF = np.zeros((6, self.height, self.width))

        self.lateralInhibitionKernel = self._DoGFilter(kernelsize=5, F=5., sigma1=1., sigma2=3.)
        self.lateralInhibitionKernel2 = self.makeLateralInhibitionFilters(size=3, sigma=1)

        self.thetas = np.arange(9) * np.pi / 4
        self.LoomingEnergy = []
        self.Spikes = []

    def __call__(self, cur_image, prev_image):
        '''Photoreceptors Layer'''
        prev_image = np.double(cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY))
        cur_image = np.double(cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY))

        cur_image = cv2.GaussianBlur(cur_image, GAUSSIAN_KERNEL_SIZE, sigmaX=1, sigmaY=1)
        prev_image = cv2.GaussianBlur(prev_image, GAUSSIAN_KERNEL_SIZE, sigmaX=1, sigmaY=1)

        photorecepters = self.HighpassFilter(cur_image, prev_image)

        Bipolar_ON = relu(photorecepters)
        Bipolar_ON = cv2.filter2D(Bipolar_ON, -1, self.lateralInhibitionKernel)
        Bipolar_OFF = negrelu(photorecepters)
        Bipolar_OFF = cv2.filter2D(Bipolar_OFF, -1, self.lateralInhibitionKernel)

        '''Bipolar Layer'''
        Bipolar_ON_Fast_Output, Bipolar_ON_Slow_Output, self.Bipolar_ON = self.TemporalFiltering2(
            Bipolar_ON, self.Bipolar_ON, A=60, tau=5, dt=0.05, K=5)
        Bipolar_OFF_Fast_Output, Bipolar_OFF_Slow_Output, self.Bipolar_OFF = self.TemporalFiltering2(
            Bipolar_OFF, self.Bipolar_OFF, A=60, tau=5, dt=0.05, K=5)

        '''Starburst Amacrine Cells'''

        Energy_ONs = []
        Energy_OFFs = []
        Energy_SUMs = []
        MotionEnergy_SUM = np.zeros_like(self.thetas)
        MotionEnergy_SUM2 = np.zeros_like(self.thetas)

        for i, theta in enumerate(self.thetas):
            if i<4:
                g1, g2 = self.makeSpatialFilters(size=5, sigma=0.3, theta=theta, lamda=4)
                Energy_ON, Energy_OFF = self.computeMotionEnergy(Bipolar_ON_Slow_Output, Bipolar_OFF_Slow_Output,
                                                                    Bipolar_ON_Fast_Output, Bipolar_OFF_Fast_Output, g1, g2)
                Energy_ONs.append(Energy_ON)
                Energy_OFFs.append(Energy_OFF)
                Energy_SUMs.append(Energy_ON + Energy_OFF)
                MotionEnergy_SUM[i] = Energy_OFF.sum() + Energy_ON.sum()
                MotionEnergy_SUM2[i] = relu(Energy_OFF).sum() + relu(Energy_ON).sum()
            else:
                Energy_ON = -Energy_ONs[i-4]
                Energy_OFF = -Energy_OFFs[i-4]
                Energy_ONs.append(Energy_ON)
                Energy_OFFs.append(Energy_OFF)
                Energy_SUMs.append(Energy_ON + Energy_OFF)
                MotionEnergy_SUM[i] = Energy_OFF.sum() + Energy_ON.sum()
                MotionEnergy_SUM2[i] = relu(Energy_OFF).sum() + relu(Energy_ON).sum()

        f1, f2 = self.makeNonDirectiveInhibitionFilters(size=5, sigma=0.3, lamda=4)
        ON_Inhibit_Fast, OFF_Inhibit_Fast, ON_Inhibit_Slow, OFF_Inhibit_Slow = self.computeInhibition(Bipolar_ON_Slow_Output, 
                                                        Bipolar_OFF_Slow_Output, Bipolar_ON_Fast_Output, Bipolar_OFF_Fast_Output, f1)

        Energy_ONs = np.asarray(Energy_ONs)
        Energy_OFFs = np.asarray(Energy_OFFs)
        Energy_SUMs = np.asarray(Energy_SUMs)
        Energy_ON_Max = np.max(Energy_ONs, 0)
        Energy_OFF_Max = np.max(Energy_OFFs, 0)
        Energy_SUM_Max = np.max(Energy_SUMs, 0)

        EstimatedDirection = np.arctan2(Energy_SUMs[2], Energy_SUMs[0])

        Vx = np.cos(EstimatedDirection)
        Vy = np.sin(EstimatedDirection)

        Dx = Vx * Energy_SUM_Max  # * Saliency_Mask
        Dy = Vy * Energy_SUM_Max  # * Saliency_Mask

        '''Direction-selective Ganglion Cells and Approach-selective Ganglion Cells'''
        PV5_ON_0 = relu(relu(Bipolar_ON_Slow_Output) - relu(OFF_Inhibit_Fast))
        PV5_ON_1 = relu(relu(Bipolar_ON_Fast_Output) - relu(OFF_Inhibit_Slow))
        PV5_OFF_0 = relu(relu(Bipolar_OFF_Slow_Output) - relu(ON_Inhibit_Fast))
        PV5_OFF_1 = relu(relu(Bipolar_OFF_Fast_Output) - relu(ON_Inhibit_Slow))
        # Lateral Inhibitation: ON-1, OFF-0.1~0.4, Radial Motion: ON-0.5, OFF-1.0
        PV5 = 1.0 * PV5_OFF_0 + 0.3*PV5_ON_1
        # PV5 = (0.5*PV5_OFF_0 + 0.0*PV5_ON_1) + (1.0*PV5_ON_0 + 0.0*PV5_OFF_1)
        # PV5 = (1.0 * PV5_OFF_0 + 0.2 * PV5_ON_1) + (0.5 * PV5_ON_0 + 0.2 * PV5_OFF_1)

        kernalE = 1. / 9 * np.ones((3, 3))
        Ce = cv2.filter2D(PV5, -1, kernalE)
        PV5 = PV5 * Ce

        # Saliency_map = cv2.GaussianBlur(PV5, (51, 51), sigmaX=8, sigmaY=8)
        Saliency_map1 = cv2.GaussianBlur(PV5, (31, 31), sigmaX=8, sigmaY=8)
        Saliency_Mask1 = (Saliency_map1 > 0.01) * 1.

        top_2_idx = MotionEnergy_SUM.argsort()[::-1][0:1]
        NewVal = Energy_SUM_Max - np.max(Energy_SUMs[top_2_idx], 0)
        Saliency_map2 = cv2.GaussianBlur(NewVal, (31, 31), sigmaX=8, sigmaY=8)
        Saliency_Mask2 = (Saliency_map2 > 0.008) * 1.

        Saliency_Mask = Saliency_Mask1 * Saliency_Mask2
        PV5_New = PV5 * Saliency_Mask

        estimated_pos, estimated_target_dv = [], []
        Saliency_Mask[0,0] = 0.51
        [IndX, IndY] = np.where(Saliency_Mask > 0.5)
        X = np.array([IndX, IndY]).T
        db = DBSCAN(eps=5, min_samples=2).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        unique_labels = set(labels)
        estimated_target_dv = []
        estimated_target_direction = []
        estimated_pos = []

        for k in zip(unique_labels):
            if k[0] >= 0:
                class_member_mask = (labels == k)
                xy = X[class_member_mask & core_samples_mask]
                pos = np.zeros_like(xy)
                pos[:, 0] = xy[:, 1]
                pos[:, 1] = xy[:, 0]
                meanpos = np.mean(pos, 0)
                sum_dx = 0.
                sum_dy = 0.
                for i in range(pos.shape[0]):
                    sum_dx += Dx[int(xy[i, 0]), int(xy[i, 1])]
                    sum_dy += Dy[int(xy[i, 0]), int(xy[i, 1])]
                mean_dx = sum_dx / pos.shape[0]
                mean_dy = sum_dy / pos.shape[0]
                dist = np.sqrt(np.sum(np.square(meanpos - np.zeros((1, 2))), 1))
                if dist > 1:
                    estimated_pos.append(meanpos)
                    estimated_target_dv.append(np.array([mean_dx, mean_dy]))
                    estimated_target_direction.append(np.arctan2(mean_dy, mean_dx))

        K = np.sum(np.abs(PV5_New * 255.))
        kf = 1. / (1 + np.exp(-K / (self.width * self.height)))

        Ksp = 10
        Tsp = 0.51
        SPI = np.floor(np.exp(Ksp * (kf - Tsp)))
        # SPI = np.exp(Ksp * (kf - Tsp))
        # SPI = np.exp(self.Ksp * (self.SFAs[-1] - self.Tsp))
        self.Spikes.append(SPI)
        Spikes = np.asarray(self.Spikes[-4:])
        SpikeFrequency = np.sum(Spikes) * 1000. / (4 * 25)
        cv2.imshow('t1', cur_image/255.)
        cv2.imshow('t2', Saliency_Mask1 * 255.)
        cv2.imshow('t3', Saliency_Mask2 * 255.)
        cv2.imshow('t4', Saliency_Mask * 255.)
        cv2.waitKey(5)

        return PV5, PV5_New, Energy_SUM_Max, Saliency_Mask1, Saliency_Mask2, SpikeFrequency, Dx, Dy, estimated_pos, estimated_target_dv, self.thetas, MotionEnergy_SUM

        # return Saliency_Mask, PV5.sum(), 0, 0, 0, 0

    def HighpassFilter(self, cur_input, pre_input):
        return (cur_input - pre_input)/255.

    def LowpassFilter(self, cur_input, pre_input, lp_t):
        return lp_t * cur_input + (1 - lp_t) * pre_input

    def _DoGFilter(self, kernelsize=5, F=1.,  sigma1=1., sigma2=1.):
        flag = np.mod(kernelsize, 2)
        if flag == 0:
            kernelsize += 1
        CenX = np.round(kernelsize / 2)
        CenY = np.round(kernelsize / 2)
        X, Y = np.meshgrid(np.arange(kernelsize), np.arange(kernelsize))
        ShiftX = X - CenX
        ShiftY = Y - CenY
        Gauss1 = 1. / (2 * np.pi * sigma1 ** 2) * np.exp(-(ShiftX * ShiftX + ShiftY * ShiftY) / (2 * sigma1 ** 2))
        Gauss2 = 1. / (2 * np.pi * sigma2 ** 2) * np.exp(-(ShiftX * ShiftX + ShiftY * ShiftY) / (2 * sigma2 ** 2))
        DoG_Filter = F * (Gauss1 - Gauss2)
        return DoG_Filter

    def TemporalFiltering2(self, cur_input, pre_input, A=60, tau=8, dt=0.04, K=5, nf=2, ns=4):
        n = pre_input.shape[0]
        dt_tau = dt / tau
        for i in range(n):
            I = cur_input if i == 0 else pre_input[i - 1]
            dy = - A * pre_input[i] + A * I
            pre_input[i] += dt_tau * dy

        out_fast = K * (pre_input[nf] - pre_input[nf + 1])
        out_slow = K * (pre_input[ns] - pre_input[ns + 1])
        return out_fast, out_slow, pre_input

    def makeSpatialFilters(self, size, sigma, theta, lamda):
        x = np.linspace(-1., 1., size)
        y = np.linspace(-1., 1., size)
        X, Y = np.meshgrid(x, y)
        theta = np.pi - theta
        rotx = X * np.cos(theta) + Y * np.sin(theta)
        roty = -X * np.sin(theta) + Y * np.cos(theta)
        kernel1 = np.exp(-(rotx ** 2 + roty ** 2) / (2 * sigma ** 2))
        kernel1 *= np.cos(2 * np.pi * rotx / lamda)

        kernel2 = np.exp(-(rotx ** 2 + roty ** 2) / (2 * sigma ** 2))
        kernel2 *= np.sin(2 * np.pi * rotx / lamda)
        return kernel1, kernel2

    def makeDirectiveInhibitionFilters(self, size, sigma, theta, r):
        flag = np.mod(size, 2)
        if flag == 0:
            size += 1
        theta = np.pi - theta
        dX = r * np.cos(theta)
        dY = r * np.sin(theta)
        X, Y = np.meshgrid(np.arange(size) - np.round(size / 2),
                           np.arange(size) - np.round(size / 2))
        ShiftX1 = X + dX
        ShiftY1 = Y + dY
        ShiftX2 = X - dX
        ShiftY2 = Y - dY
        Gauss1 = 1. / (2 * np.pi * sigma ** 2) * np.exp(-(ShiftX1 * ShiftX1 + ShiftY1 * ShiftY1) / (2 * sigma ** 2))
        Gauss2 = 1. / (2 * np.pi * sigma ** 2) * np.exp(-(ShiftX2 * ShiftX2 + ShiftY2 * ShiftY2) / (2 * sigma ** 2))
        kernel = Gauss1 - Gauss2
        return kernel

    def makeNonDirectiveInhibitionFilters(self, size, sigma, lamda):
        x = np.linspace(-1., 1., size)
        y = np.linspace(-1., 1., size)
        (Y, X) = np.meshgrid(y, x)

        kernel1 = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        kernel1 *= np.cos(2 * np.pi * (X ** 2 + Y ** 2) / lamda)

        kernel2 = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        kernel2 *= np.sin(2 * np.pi * (X ** 2 + Y ** 2) / lamda)
        return kernel1, kernel2

    def makeLateralInhibitionFilters(self, size, sigma):
        x = np.linspace(-1., 1., size)
        y = np.linspace(-1., 1., size)
        X, Y = np.meshgrid(x, y)
        kernel = 1 - np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
        return kernel

    def computeMotionEnergy(self, Bipolar_ON_Slow_Output, Bipolar_OFF_Slow_Output, Bipolar_ON_Fast_Output, Bipolar_OFF_Fast_Output, f1, f2):
        A1_ON = cv2.filter2D(Bipolar_ON_Slow_Output, -1, f1)
        B1_ON = cv2.filter2D(Bipolar_ON_Slow_Output, -1, f2)
        A2_ON = cv2.filter2D(Bipolar_ON_Fast_Output, -1, f1)
        B2_ON = cv2.filter2D(Bipolar_ON_Fast_Output, -1, f2)

        A1_OFF = cv2.filter2D(Bipolar_OFF_Slow_Output, -1, f1)
        B1_OFF = cv2.filter2D(Bipolar_OFF_Slow_Output, -1, f2)
        A2_OFF = cv2.filter2D(Bipolar_OFF_Fast_Output, -1, f1)
        B2_OFF = cv2.filter2D(Bipolar_OFF_Fast_Output, -1, f2)

        Energy_ON = A1_ON * B2_ON - A2_ON * B1_ON
        Energy_OFF = A1_OFF * B2_OFF - A2_OFF * B1_OFF

        return Energy_ON, Energy_OFF

    def computeInhibition(self, Bipolar_ON_Slow_Output, Bipolar_OFF_Slow_Output, Bipolar_ON_Fast_Output, Bipolar_OFF_Fast_Output, f1):
        ON_Inhibit_Fast = cv2.filter2D(Bipolar_ON_Fast_Output, -1, 1-f1)
        OFF_Inhibit_Fast= cv2.filter2D(Bipolar_OFF_Fast_Output, -1, 1-f1)
        ON_Inhibit_Slow = cv2.filter2D(Bipolar_ON_Slow_Output, -1, 1-f1)
        OFF_Inhibit_Slow= cv2.filter2D(Bipolar_OFF_Slow_Output, -1, 1-f1)

        return ON_Inhibit_Fast, OFF_Inhibit_Fast, ON_Inhibit_Slow, OFF_Inhibit_Slow

    def computeLoomingEnergy(self, Bipolar_ON_Fast, Bipolar_ON_Slow, Bipolar_OFF_Fast, Bipolar_OFF_Slow, kernel):
        Bipolar_ON_Fast_Inh = cv2.filter2D(Bipolar_ON_Fast, -1, kernel)

        Bipolar_OFF_Fast_Inh = cv2.filter2D(Bipolar_OFF_Fast, -1, kernel)
        Bipolar_ON = Bipolar_ON_Slow - Bipolar_OFF_Fast_Inh
        Bipolar_OFF = Bipolar_OFF_Slow - Bipolar_ON_Fast_Inh

        Scells = 0.4*Bipolar_ON + Bipolar_OFF

        kernalE = 1. / 9 * np.ones((3, 3))
        Ce = cv2.filter2D(Scells, -1, kernalE)
        # omega = 0.01 + np.sum(abs(Ce))
        RGCcells = Scells * Ce

        Energy = RGCcells.sum()

        # cv2.imshow('t2', RGCcells*255)
        # cv2.waitKey(5)
        return RGCcells, Energy

def saveData(filename, data):
    output = open(filename, 'wb')
    pickle.dump(data, output)
    output.close()


def loadData(filename):
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", help="Training step size", type=int, default=130)
    parser.add_argument('--run_name', type=str, default=time.strftime('%Y%m%d%H%M%S', time.localtime()),
                        help='Name for the run')
    parser.add_argument('--log_dir', type=str, default='logs', help='After how many epochs to print logs')
    
    args = parser.parse_args()
    
    # Add timestamp to log path
    args.log_dir = os.path.join(args.log_dir, '%s' % args.run_name)
    
    # Add model name to log path
    args.log_dir = args.log_dir + '_ASNN_RadialMotion_speed_0.05'
    
    # Create log path
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    
    timesteps = args.timesteps
    
    # Create task content
    # content = ObjectLoomingContent()
    # content = SmallObjectDetectionContent()
    # content = ObjectMovingContent()
    content = ObjectLoomingBackgroundContent()
    
    retina = Retina(128, 128)

    # Environment
    env = Environment(content)
    obs = env.reset()
    pre_image = obs['screen']
    Energy_Ls = []
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size':13}
    for i in range(timesteps):
        action = np.zeros(2)
        # Foward environment one step
        obs, reward, done, _ = env.step(action)
        env.render()
    
        cur_image, angle = obs['screen'], obs['angle']
        # Choose action by the agent's decision
        pv5, pv5_new, motionenergy, mask1, mask2, energy_looming, Dx, Dy, estimated_pos, estimated_direction, thetas, motionenergy_sum = retina(cur_image, pre_image)
    
        pre_image = cur_image.copy()
        Energy_Ls.append(energy_looming)
    
        # Display the estimated optic flow
        if i == 32:
            print(i)
            X = np.arange(0, 128)
            Y = np.arange(0, 128)
            sample = 2
            IndexX = np.arange(0, 128, sample)
            IndexY = np.arange(0, 128, sample)
            l = sample / 2.
    
            fig = plt.figure(figsize=(12, 2))
            ax0 = fig.add_subplot(1, 6, 1)
            ax0.imshow(cur_image, cmap=cmap, interpolation="nearest")
            ax0.quiver(X[::sample], Y[::sample], Dx[::sample, ::sample], -Dy[::sample, ::sample], scale=1.0,
                       color='r', angles='xy', pivot='middle', headwidth=3, headlength=3, linewidth=5)
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_title('(a)', font)
    
            ax1 = fig.add_subplot(1, 6, 2)
            ax1.imshow(motionenergy, cmap = cmap, interpolation = "nearest")
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title('(b)', font)
    
            ax2 = fig.add_subplot(1, 6, 3)
            ax2.imshow(pv5, cmap=cmap, interpolation="nearest")
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title('(c)', font)
    
            ax3 = fig.add_subplot(1, 6, 4)
            ax3.imshow(mask1, cmap=cm.gray, interpolation="nearest")
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title('(d)', font)
    
            ax4 = fig.add_subplot(1, 6, 5)
            ax4.imshow(mask2, cmap=cm.gray, interpolation="nearest")
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_title('(e)', font)
    
            ax5 = fig.add_subplot(1, 6, 6)
            ax5.imshow(pv5_new, cmap=cmap, interpolation="nearest")
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.set_title('(f)', font)
    
            # if estimated_pos != []:
            #     estimated_pos = np.asarray(estimated_pos)
            #     estimated_direction = np.asarray(estimated_direction)
            #     ax1.plot(estimated_pos[:,0], estimated_pos[:,1], 'bo')
            #     ax1.quiver(estimated_pos[:,0], estimated_pos[:,1], estimated_direction[:,0]*10, -estimated_direction[:,1]*10, scale=1.0, color='b', angles='xy', headwidth=3,
            #                 headlength=3)
            # plt.savefig('logs/approach_background.pdf', bbox_inches='tight', dpi=400)
            plt.show()
    
    # LoomingEnergys = {"LoomingEnergy": Energy_Ls}
    # saveData(args.log_dir + '/LoomingEnergy.pkl', LoomingEnergys)
    
    fig = plt.figure(figsize=(12,3))
    t = np.arange(len(Energy_Ls))
    x = np.ones(len(Energy_Ls))-0.5
    y = np.asarray(Energy_Ls)
    plt.plot(t, y)
    plt.fill_between(t, x, y, interpolate=True, color='green', alpha=0.5)
    plt.show()
    cv2.destroyAllWindows()