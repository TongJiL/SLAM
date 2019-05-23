#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:36:58 2019

@author: luodi
"""

import numpy as np
import map_utils
import matplotlib.pyplot as plt; plt.ion()

class Particle_filter(object):
    def __init__(self, MAP, N):
        '''
        initialize particles
        N: number of the particles
        '''
        self.particles = np.zeros((N,4))
#        self.particles[:,0] = (MAP['xmax']-MAP['xmin'])*np.random.rand(N)+MAP['xmin']
#        self.particles[:,1] = (MAP['ymax']-MAP['ymin'])*np.random.rand(N)+MAP['ymin']
#        self.particles[:,2] = 2*np.pi*np.random.rand(N)
        self.particles[:,3] = 1/N
        
    def get_particles_map(self, MAP, particles):
        '''
        get the particles position on the map 
        '''
        x = particles[:,0]
        y = particles[:,1]
        x_ceil = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        y_ceil = np.ceil((y - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        
        MAP['particles'] = MAP['map']
        MAP['particles'][x_ceil, y_ceil] = 1
        
        return MAP
    
    def update_particles(self, MAP, particles, lidar_range, angles):
        '''
        From current location of the particles and the lidar scan
        update the new map
        '''
        from TEST import Mapping
        mp = Mapping()
        angles, ranges = mp.rm_ld(angles, lidar_range)
        (x_l, y_l) = mp.pol2car(angles, ranges)
        h_l = mp.car2hom(x_l, y_l)
        h_b = mp.trans_l2b(h_l)
        
        cor_max = []
        
        for i in range(particles.shape[0]):
            x_p = particles[i,0]
            y_p = particles[i,1]
            theta_p = particles[i,2]
            
            hp_w = mp.trans_b2w(h_b, x_p, y_p, theta_p)
            xp_w, yp_w = mp.hom2car(hp_w)
            
            # convert position in the map frame here 
            Y = np.stack((xp_w,yp_w))
            
            x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
            y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

            x_range = np.arange(-0.2,0.2+0.05,0.05)
            y_range = np.arange(-0.2,0.2+0.05,0.05)
            
            cor = abs(map_utils.mapCorrelation(MAP['map'], x_im, y_im, Y, x_range, y_range))


            cor_max.append(np.max(cor))
     
        return cor, cor_max
    
    
    def resample(self,weights):
        n = len(weights)
        indices = []
        # 求出离散累积密度函数(CDF)
        C = [0.] + [sum(weights[:i+1]) for i in range(n)] 
        # 选定一个随机初始点
        u0, j = np.random.random(), 0
        for u in [(u0+i)/n for i in range(n)]: # u 线性增长到 1
            while u > C[j]: # 碰到小粒子，跳过
                j+=1
            indices.append(j-1)  # 碰到大粒子，添加，u 增大，还有第二次被添加的可能
        return indices # 返回大粒子的下标

  
    def trjandmap(self,mppp,x_b,y_b):
        for i in range(mppp.shape[0]):
            for j in range(mppp.shape[1]):
                if mppp[i][j]>-mppp[400][400]:
                    mppp[i][j]=0
                else:
                    mppp[i][j]=1      
        
        x_b = np.array(x_b)  
        x_tr = np.ceil((x_b - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1  
        y_b = np.array(y_b)  
        y_tr = np.ceil((y_b - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        
        mppp[x_tr,y_tr]=0 
        
        return mppp
    
   
if __name__ == '__main__':
    
  dataset = 23
  
  with np.load("Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load("Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
#  with np.load("Kinect%d.npz"%dataset) as data:
#    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
#    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
#    
  from TEST import Mapping, Trajectory
  
  mp = Mapping()
  tr = Trajectory()
  
  '''.....ignore the data when encoder has no value.............'''
#  encoder_stamps = encoder_stamps[384:]
#  encoder_counts = encoder_counts[:,384:]
  lidar_stamps = lidar_stamsp
#  lidar_ranges = lidar_ranges[:,384:]
#  
 
  
  '''.................init the map of 801x801.......................'''
  

  MAP = mp.init_map()
  N=100
  
  pf = Particle_filter(MAP, N)
  particles = pf.particles
  
    
  angles = np.linspace(lidar_angle_min, lidar_angle_max, 1081)
  ranges = lidar_ranges[:,0]
  
#  MAP = mp.mapping(MAP, angles, ranges, x_0, y_0, theta_0)
#  plt.imshow(MAP['map'])
#  plt.show()
#  
#  MAP, cor, cor_max = pf.update_particles(MAP, particles, ranges, angles)
# 
#  z = particles[:,3] * cor_max
#  
#  particles[:,3] = (np.exp(z)/sum(np.exp(z)))
#  
#  N_eff = 1/sum(particles[:,3]*particles[:,3])
#  
  yaw_row = imu_angular_velocity[2,:]
  t_max = max(lidar_stamps[-1], imu_stamps[-1], encoder_stamps[-1])
  t_min = min(encoder_stamps[0],lidar_stamps[0])
    
  t = t_min
  i = 1
  j = 0
  q = 0
  z = 0
  v_robt = 0
  x_b = []
  y_b = []
  v_robt = 0
  x_best = 0
  y_best = 0
  theta_best = 0
  N_threshold = 5
  

  while t<encoder_stamps[3750]:

      if encoder_stamps[i]<t:
          
          move_count = encoder_counts[:,i]
          time_gap = encoder_stamps[i] - encoder_stamps[i-1]
            
          v_robt = (sum(move_count)/4*0.0022)/time_gap
          c=0
          imu=[]
          time_count=[]
          if imu_stamps[j]>encoder_stamps[i]:
              omega = 0
          else:
             while imu_stamps[j]<encoder_stamps[i-1]:
                 j+=1    
             
             while  encoder_stamps[i-1]< imu_stamps[j] <= encoder_stamps[i]:
                 time_count.append(imu_stamps[j])
                 imu.append(yaw_row[j])
                 j+=1
                 c+=1
             omega = sum(imu)/c
          i+=1 
          
          for k in range(N):
              x_0 = particles[k,0]
              y_0 = particles[k,1]
              theta_0 = particles[k,2]
              x, y, theta = tr.new_pos(x_0, y_0, v_robt, theta_0, omega, time_gap)
              particles[k,0] = x
              particles[k,1] = y
              particles[k,2] = theta
              
#          pf.get_particles_map(MAP, particles)
  
#          m_p = MAP['particles']
#          plt.imshow(m_p,cmap='hot')
#          plt.show()
          # compute the map correlation    
          cor, cor_max = pf.update_particles(MAP, particles, lidar_ranges[:,q], angles)
          cor_max = cor_max/np.max(cor_max)*5
          
          z = particles[:,3] * np.exp(cor_max)
  
          particles[:,3] = (np.exp(z)/sum(np.exp(z)))
          
          ind = np.argmax(particles[:,3])
          
          x_best = particles[ind,0]
          y_best = particles[ind,1]
          theta_best = particles[ind,2]
          x_b.append(x_best)
          y_b.append(y_best)
          #print(x_best,y_best)
          
          # resample
          indices = np.zeros([1,N])
          N_eff = 1/np.sum(particles[:,3]**2)
          if N_eff <= N_threshold:
              indices = pf.resample(particles[:,3])
              particles[:] = particles[indices]
              particles[:,3] = particles[:,3]/np.sum(particles[:,3])

      if lidar_stamps[q]<t:
          MAP = mp.mapping(MAP, angles, lidar_ranges[:,q], x_best, y_best, theta_best)
          q+=1
          
      t+=0.001      
        
  plt.imshow(MAP['map'])
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x_b, y_b)
  plt.show()      
        
#  mppp = MAP['map']
#    
#  mapap = pf.trjandmap(mppp, x_b, y_b)
#  plt.imshow(mapap, cmap='hot')     
        
  mp.plot_map(MAP)       
#        
#  pf.get_particles_map(MAP, particles)
#  
#  m_p = MAP['particles']
#  plt.imshow(m_p)
#  plt.show()      
 
  plt.scatter(particles[:,0],particles[:,1])
  plt.show()
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        