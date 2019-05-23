#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:36:58 2019

@author: luodi
"""

import numpy as np
import map_utils
import matplotlib.pyplot as plt; plt.ion()
import os
import cv2 as cv
from PIL import Image

class Texture_mapping(object):
    '''
    This class contains the information to get the RGB of the corresponding position
    '''
    
    def car2hom(self, x, y):
        '''
        transform the cartesian coord to homogeneous coord
        '''
        zeros = np.zeros((x.shape))
        ones = np.ones((y.shape))
        hom = np.vstack((x,y,zeros,ones))
        return hom
    
    
#    def dim3t2(self, dis_img):
#        '''
#        transform the 3 dim matrix to 2 dim
#        '''
#        dep = np.zeros([dis_img.shape[0],dis_img.shape[1]])
#        for i in range(dep.shape[0]):
#            for j in range(dep.shape[1]):
#                dep[i][j]=dis_img[i][j][0]
#        return dep
#    
    def get_rgb(self, dep, rgb_img):
        '''
        use the current depth camera to get the rgb of each pixel
        '''
        a = []
        b = []
        c = []
        depth = []
        for i in range(dep.shape[0]):
            for j in range(dep.shape[1]):
                d = dep[i][j]
                dd = (-0.00304*d+3.31)
                rgbi = int(round((i*526.37+dd*(-4.5*1750.46)+19276)/585.051))
                rgbj = int(round((j*526.37+16662)/585.051))
                a.append(i)
                b.append(j)
                if rgbi<480 and rgbj<640:
                    c.append(rgb_img[rgbi][rgbj][:])
                depth.append(1.03/dd)
        a = np.array(a)
        b = np.array(b)
        rgb = np.array(c)
        depth = np.array(depth)
        depth = depth.reshape(1,len(depth))
        rgb_df = np.vstack((b, a))
        return rgb_df, rgb, depth
        
    
    def trans_px2dp(self, rgb_df, depth):
        '''
        transform the depth pixels to depth frame
        '''
        ones = np.ones((rgb_df.shape[1]))
        pixel = np.vstack((rgb_df, ones))
        A = np.matrix([[585.05108211, 0, 242.94140713],
                       [0, 585.05108211, 315.83800193],
                       [0, 0, 1]])
        B = np.matrix([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        cor = np.array(np.linalg.pinv(B)@np.linalg.pinv(A)@pixel)
        cor[0,:] = cor[0,:]*depth
        cor[1,:] = cor[1,:]*depth
        cor[2,:] = cor[2,:]*depth
        return cor
    
    def trans_dp2b(self, cor):
        '''
        transform the depth camera frame to the body frame
        '''
        h_db = np.zeros([cor.shape[0], cor.shape[1]])
        h_db[0,:] = cor[2,:]-0.18
        h_db[1,:] = -cor[0,:]-0.005
        h_db[2,:] = -cor[1,:]-0.36
        h_db[3,:] = np.ones((1,cor.shape[1]))
       
        return h_db
    
    def hom2car(self, h):
        '''
        transform hom coord to car coord
        '''
        x = np.asarray(h[0]).reshape(-1)
        y = np.asarray(h[1]).reshape(-1)
        z = np.asarray(h[2]).reshape(-1)
        return x, y, z
    
    def get_color(self, d_img, r_img, x_best, y_best, theta_best, MAP):
        '''
        From the image and the current position, get the color of the MAP
        '''
        dep = np.array(d_img)
        rgb_df, rgb, depth = self.get_rgb(dep, r_img)
        cor = self.trans_px2dp(rgb_df, depth)
        h_db = self.trans_dp2b(cor)
        h_w = mp.trans_b2w(h_db, x_best, y_best, theta_best)
        x,y,z = self.hom2car(h_w)
        for i in range(len(x)):
            if z[i]>0.254:
                x[i]=y[i]=0
        x_ce = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        y_ce = np.ceil((y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        ind = np.logical_and(np.logical_and(np.logical_and((x_ce > 1), (y_ce > 1)), (x_ce < MAP['sizex'])), (y_ce < MAP['sizey']))
        
        for i in range(len(ind)):
            if ind[i] == False:
                x_ce[i]=y_ce[i]=1000
                
        for j in range(len(rgb)):
            MAP['rgb'][x_ce[j]][y_ce[j]] = rgb[j]
            
        return MAP
     
          

  
   
if __name__ == '__main__':
    
  dataset = 20
  
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
  
  with np.load("Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    
  disparity = os.listdir('dataRGBD/Disparity20/')
  RGB = os.listdir('dataRGBD/RGB20/')
  dis_img = []
  for i in range(len(disparity)):
      dis_img.append(Image.open('dataRGBD/Disparity20/%s'%disparity[i]))
      
  rgb_img = []
  for i in range(len(RGB)):
      rgb_img.append(plt.imread('dataRGBD/RGB20/%s'%RGB[i]))
   
    
    
  from TEST import Mapping, Trajectory
  from particles import Particle_filter
  
  mp = Mapping()
  tr = Trajectory()
  te = Texture_mapping()  
    
  
  '''.....ignore the data when encoder has no value.............'''
#  encoder_stamps = encoder_stamps[384:]
#  encoder_counts = encoder_counts[:,384:]
#  lidar_stamps = lidar_stamsp[384:]
#  lidar_ranges = lidar_ranges[:,384:]
#    

  '''.................init the map of 801x801.......................'''
  

  MAP = mp.init_map()
  N=100
  
  pf = Particle_filter(MAP, N)
  particles = pf.particles
  
    
  angles = np.linspace(lidar_angle_min, lidar_angle_max, 1081)
  ranges = lidar_ranges[:,0]

  yaw_row = imu_angular_velocity[2,:]
  t_max = max(lidar_stamsp[-1], imu_stamps[-1], encoder_stamps[-1])
  t_min = min(encoder_stamps[0],lidar_stamsp[0])
    
  t = t_min
  i = 1
  j = 0
  q = 0
  z = 0
  r = 0
  d = 0
  v_robt = 0
  x_b = []
  y_b = []
  v_robt = 0
  x_best = 0
  y_best = 0
  theta_best = 0
  N_threshold = 5
  count = 1

  while t<encoder_stamps[3500]:

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

      if lidar_stamsp[q]<t:
          MAP = mp.mapping(MAP, angles, lidar_ranges[:,q], x_best, y_best, theta_best)
          while disp_stamps[d] < lidar_stamsp[q]:
              d+=1
          while rgb_stamps[r] < disp_stamps[d]:
              r+=1
          d_img = dis_img[d]
          r_img = rgb_img[r]
          MAP = te.get_color(d_img, r_img, x_best, y_best, theta_best, MAP)
          print(count)
          count+=1
          q+=1
          
      t+=0.001      
        
  plt.imshow(MAP['map'])
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x_b, y_b)
  plt.show()      
    
        
  mp.plot_map(MAP)            
 
  plt.scatter(particles[:,0],particles[:,1])
  plt.show()
   
  plt.imshow(MAP['rgb'])  
 
     
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        