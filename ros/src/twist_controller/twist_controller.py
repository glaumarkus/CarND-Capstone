from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
#MAX_THROTTLE = 0.5

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, 
                    decel_limit, accel_limit, wheel_radius, wheel_base, 
                    steer_ratio, max_lat_accel, max_steer_angle):
        
        self.yaw_controller = YawController(wheel_base, steer_ratio,
                    0.1, max_lat_accel, max_steer_angle)
        
        '''
        # experimental parameters 
        kp = 0.3
        ki = 0.1
        kd = 0.02
        mn = 0. # minimum throttle
        mx = 0.2 # maximum throttle 
        self.throttle_controller = PID(kp,ki,kd,mn,mx)
        '''
        kp = 0.5
        ki = 0.0001
        kd = 0.15
        mn = decel_limit # minimum throttle
        mx = 0.5 # maximum throttle 
        self.throttle_controller = PID(kp,ki,kd,mn,mx)
        
        # to control the noisy messages in current_velocity via a provided low pass filter
        tau = 0.05
        ts = 0.02
        self.velocity_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass=vehicle_mass
        self.brake_deadband=brake_deadband
        self.decel_limit=decel_limit
        self.accel_limit=accel_limit
        self.wheel_radius=wheel_radius

        self.last_time = rospy.get_time()
        self.last_throttle = 0
        self.last_brake = 100
        
    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):

        # check if car is in manual mode
        if not dbw_enabled:
            self.throttle_controller.reset() 
                # disconnect the PID controllet to avoid error accumulating for the Integral term
                # to avoid erratic behaviours at re-insertion of the controller
            return 0., 0., 0.
        
        # get current velocity
        current_vel = self.velocity_lpf.filt(current_vel)

        ## get the comands
        steer = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)

        velocity_error = linear_vel-current_vel
       # self.last_velocity = current_vel

        ## update internal time
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        # get throttle
        aceeleration = self.throttle_controller.step(velocity_error, sample_time)
        #the square difference between the proposed velocity (linear_vel) and the current velocity (current_vel) 
        #divided by 2 times the distance of 30 meters. 
        #The positive value that returns this algorithm are the throttle values of the car.
        
        smooth_acc = ((linear_vel*linear_vel)-(current_vel*current_vel))/(2*30)
        
        if smooth_acc >= 0:
            throttle = smooth_acc 
        else:
            throttle = 0

        if throttle > 0.6:
            throttle = 0.6       

        #smoothing throttle acceleration and deceleration    
        if (throttle > 0.025) and (throttle - self.last_throttle) > 0.005:
            throttle = max((self.last_throttle + 0.0025), 0.005)
       
        if throttle > 0.025 and (throttle - self.last_throttle) < -0.05:
            throttle = self.last_throttle - 0.05     

        self.last_throttle = throttle
        
        # get brake
        brake = 0
        
        if linear_vel == 0 and current_vel < 0.1: # 0.1 m/s is the minimum velocity
            throttle = 0
            brake = 700 # N*m - amount necessary to hold the car in place while stopped
                # like if we're waiting at a traffic light
                # tha corresponding acceleration is around 1m/s^2
        elif throttle < 0.025 and velocity_error < 0.: #if we're stopping/decelerating
            throttle = 0
            decel = max((smooth_acc * 5), self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N*m
            #smoothing brake

            if brake > 100 and (brake - self.last_brake) > 20:
                brake = max((self.last_brake + 20), 100)

        if brake > 20 and (brake - self.last_brake) > 20:
            brake = max((self.last_brake + 20), 20)

        #rospy.loginfo('brake: %f', brake)
        #rospy.loginfo('trottle: %f', throttle)
        self.last_brake = brake
        return throttle, brake, steer
