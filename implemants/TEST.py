import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
import math

class Mapping(object):
    '''
    This class is for mapping part, refer as mp.
    '''
    
    def pol2car(self, phi, rad):
        '''
        convert polar coord to cartesian coord
        param: phi, angle
        type: np.ndarray
        param: rad, radius
        type: np.ndarray
        rtype: (x, y) cartesian coord, np.ndarray
        '''
        x = rad * np.cos(phi)
        y = rad * np.sin(phi)
        return(x,y)
    
    def car2hom(self, x, y):
        '''
        transform the cartesian coord to homogeneous coord
        '''
        zeros = np.zeros((x.shape))
        ones = np.ones((y.shape))
        hom = np.vstack((x,y,zeros,ones))
        return hom
    
    def hom2car(self, h):
        '''
        transform hom coord to car coord
        '''
        x = np.asarray(h[0]).reshape(-1)
        y = np.asarray(h[1]).reshape(-1)
        
        return x, y
    
      
    def trans_l2b(self, h_l):
        '''
        transform lidar frame to boay frame
        param:h_l homogeneous coord on lidar frame
        type: np.matrix
        arg: h_b homogeneous coord on body frame
        type: np.matrix
        '''
        M_b2l = np.matrix([[1, 0, 0, 133.23e-3],
                           [0, 1, 0, 0],
                           [0, 0, 1, 514.35e-3],
                           [0, 0, 1, 1]])
        
        h_b = M_b2l * h_l
        
        return h_b
      
    def trans_b2w(self, h_b, x, y, theta):
        '''
        transform from body frame to wolrd frame
        param: x, y, theta, current state, float
        param: h_b, hom coord of body frame, np.matrix
        rtype: h_w, world frame, np.martix
        '''
        M_b2w = np.matrix([[np.cos(theta), -np.sin(theta), 0, x],
                           [np.sin(theta), np.cos(theta), 0, y],
                           [0,             0,             1, 0],
                           [0,             0,             1, 0]])
        h_w = M_b2w * h_b
        return h_w
  
    # The lidar function
    def show_lidar(self, ranges):
      angles = np.arange(-135,135.25,0.25)*np.pi/180.0
      ax = plt.subplot(111, projection='polar')
      ax.plot(angles, ranges)
      ax.set_rmax(10)
      ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
      ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
      ax.grid(True)
      ax.set_title("Lidar scan data", va='bottom')
      plt.show()
      
    def rm_ld(self, angles, ranges, lidar_range_min=0.1, lidar_range_max=30):
        '''
        remove scan points that are too close or too far
        '''
        if angles.shape[0] != ranges.shape[0]:
            raise Exception('angles and ranges do not match')
        
        i = angles.shape[0] - 1
        while i>0:
            if ranges[i] < lidar_range_min or ranges[i] > lidar_range_max:
                angles = np.delete(angles, i)
                ranges = np.delete(ranges, i)
                
            i-=1
            
        return angles, ranges
    
    
    
    def bresenham2D(self, sx, sy, ex, ey):
        '''
          Bresenham's ray tracing algorithm in 2D.
          Inputs:
        	  (sx, sy)	start point of ray
        	  (ex, ey)	end point of ray
        '''
        sx = int(round(sx))
        sy = int(round(sy))
        ex = int(round(ex))
        ey = int(round(ey))
        dx = abs(ex-sx)
        dy = abs(ey-sy)
        steep = abs(dy)>abs(dx)
        if steep:
            dx,dy = dy,dx # swap 
        
        if dy == 0:
            q = np.zeros((dx+1,1))
        else:
            q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
        if steep:
            if sy <= ey:
                  y = np.arange(sy,ey+1)
            else:
                  y = np.arange(sy,ey-1,-1)
            if sx <= ex:
                x = sx + np.cumsum(q)
            else:
                x = sx - np.cumsum(q)
        else:
            if sx <= ex:
                  x = np.arange(sx,ex+1)
            else:
                x = np.arange(sx,ex-1,-1)
            if sy <= ey:
                y = sy + np.cumsum(q)
            else:
                y = sy - np.cumsum(q)
        return np.vstack((x,y))
    
    def init_map(self):
        '''
        Initialize a map for the first use
        '''
        # init MAP
        MAP = {}
        MAP['res']   = 0.05 #meters
        MAP['xmin']  = -25  #meters
        MAP['ymin']  = -25
        MAP['xmax']  =  25
        MAP['ymax']  =  25 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
        MAP['rgb'] = np.zeros((MAP['sizex'],MAP['sizey'],3),dtype=np.float32) 
        return MAP
        
        
    def mapping(self, MAP, lidar_angle, lidar_range, x, y, theta):
        '''
        Mapping the new particles
        param: m: the old map 
               lidar angle and range of this time step
               x,y,theta: current robot pose
        return: updated log-odds map
        '''   
        # import the map from last step
        #m = m.astype(np.float32)
        MAP['map'] = MAP['map'].astype(np.float32)
#        m_res   = 0.05 #meters
#        m_xmin  = -20  #meters
#        m_ymin  = -20
#        m_xmax =  20
#        m_ymax =  20
#        m_sizex  = int(np.ceil((m_xmax - m_xmin) / m_res + 1)) #cells
#        m_sizey  = int(np.ceil((m_ymax - m_ymin) / m_res + 1))
        
        MAP['res']   = 0.05 #meters
        MAP['xmin']  = -25  #meters
        MAP['ymin']  = -25
        MAP['xmax']  =  25
        MAP['ymax']  =  25 
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))

        # remove the point that are too far or too close
        angles, ranges = self.rm_ld(lidar_angle, lidar_range)
          
        # Transform to the world frame
        (x_l, y_l) = self.pol2car(angles, ranges)
        h_l = self.car2hom(x_l, y_l)
        h_b = self.trans_l2b(h_l)
        h_w = self.trans_b2w(h_b, x, y, theta)
        x_w, y_w = self.hom2car(h_w)
        
        # bresenham2D
        sx = np.ceil((x - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        sy = np.ceil((y - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        ex = np.ceil((x_w - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
        ey = np.ceil((y_w - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
        
        indGood = np.logical_and(np.logical_and(np.logical_and((ex > 1), (ey > 1)), (ex < MAP['sizex'])), (ey < MAP['sizey']))
        for i in range(len(indGood)):
            if indGood[i] == False:
                ex[i]=ey[i]=1200
        MAP['map'][ex,ey]+= np.log(4)
     
        x_r=[]
        y_r=[]
        for i in range(len(ex)):
            x, y = self.bresenham2D(sx, sy, ex[i], ey[i])
            for j in range(len(x)):
                x_r.append(x[j])
                y_r.append(y[j])

        x_r = np.array(x_r)
        y_r = np.array(y_r)
        
        # convert from meters to cells
        xis = np.ceil(x_r).astype(np.int16)-1
        yis = np.ceil(y_r).astype(np.int16)-1
        indG = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
        for i in range(len(indG)):
            if indG[i] == False:
                xis[i]=yis[i]=1200
        MAP['map'][xis,yis]+= np.log(0.25)
        return MAP
    
    def plot_map(self, MAP):
        
        mp = MAP['map']
        mp = 1-1/(1+np.exp(mp))
        
        plt.imshow(mp, cmap='hot')
        plt.show()


class Trajectory(object):
    '''
    This class is to caculate the trajectory of the robot.
    '''
    def variance(self, encoder_counts, encoder_stamps, ):
        '''
        Caculate the difference between each step as the input of the def distance
        '''
        move_counts = np.zeros((4,4955))
        time_gap = np.zeros((4955))
        for i in range(0, 4955):
            move_counts[:,i] = encoder_counts[:, i+1]
            time_gap[i] = encoder_stamps[i+1] - encoder_stamps[i]
        return move_counts, time_gap
    
    
    def new_pos(self, x, y, v_robt, theta, omega, ti):
        '''
        Caculate the x_new, y_new based on the encoder
        param: x, y, theta: robot current position
               e_c, e_s: counts of the wheel
        return: x_new, y_new, theta_new
        '''
        yaw = omega*ti
        v_robt = np.random.normal(v_robt, 0.05)
        dis_robt = v_robt*ti
        if omega != 0:
            omega = np.random.normal(v_robt, 0.1)
            x_new = x + dis_robt*np.sin(omega*ti/2)/(omega*ti/2)*np.cos(theta+(omega*ti/2))
            
            y_new = y + dis_robt*np.sin(omega*ti/2)/(omega*ti/2)*np.sin(theta+(omega*ti/2))
        else:
            x_new = x + dis_robt*np.cos(theta)
            
            y_new = y + dis_robt*np.sin(theta)
        
        theta_new = theta + yaw
              
        return x_new, y_new, theta_new
      
  
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
    
    
  mp = Mapping()
  
  #covert polar coord to car coord
  angles = np.linspace(lidar_angle_min, lidar_angle_max, 1081)
  ranges = lidar_ranges[:,384]
  
  (x_l, y_l) = mp.pol2car(angles, ranges)

  mp.show_lidar(ranges)
  
  #transform to body frame
  h_l = mp.car2hom(x_l, y_l)
  h_b = mp.trans_l2b(h_l)
  
  #transform to world frame
  h_w = mp.trans_b2w(h_b, 0, 0, 0)
  x_w, y_w = mp.hom2car(h_w)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x_w, y_w)
  plt.show()
  
  '''.....ignore the data when encoder has no value.............'''
#  encoder_stamps = encoder_stamps[384:]
#  encoder_counts = encoder_counts[:,384:]
#  lidar_stamps = lidar_stamsp[384:]
#  lidar_ranges = lidar_ranges[:,384:]
#  
  
  '''.................init the map of 801x801.......................'''
  
  theta_0 = 0
  x_0 = 0
  y_0 = 0
  MAP = mp.init_map()
  
  '''............trajectory...................'''
  
  tr = Trajectory()

 
  yaw_row = imu_angular_velocity[2,:]
  t_max = max(lidar_stamsp[-1], imu_stamps[-1], encoder_stamps[-1])
  t_min = min(lidar_stamsp[0], imu_stamps[0], encoder_stamps[0])
    
  t = t_min
  i = 1
  j = 0
  k = 0
  v_robt = 0
  x_t = []
  y_t = []
  v_robt = 0
  while t<encoder_stamps[2]:
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
#             if len(time_count)>1:
#                 ti = time_count[-1]-time_count[0]
#             else:
#                 ti = imu_stamps[j-1]-imu_stamps[j-2]

          i+=1 
          x, y, theta = tr.new_pos(x_0, y_0, v_robt, theta_0, omega, time_gap)
          x_0 = x
          y_0 = y
          theta_0 = theta
          x_t.append(x)
          y_t.append(y)
          #print(x,y)

      if lidar_stamsp[k]<t:
          MAP = mp.mapping(MAP, angles, lidar_ranges[:,k], x_0, y_0, theta_0)
          k+=1
          
      t+=0.001

  #mp.plot_map(MAP)
  plt.imshow(MAP['map'])
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x_t, y_t)
  plt.show()









































  
  
  