from pytocl.driver import Driver
from pytocl.car import State, Command
import numpy as np
from Normalize import Normalize
import tensorflow as tf

Ndata = Normalize()
saver1 = tf.train.import_meta_graph("./model_accbrk/model_accbrk.meta")
saver2 = tf.train.import_meta_graph("./model_steer/model_steer.meta")

class MyDriver(Driver):
    # Override the `drive` method to create your own driver
    def drive(self, carstate: State) -> Command:
        command = Command()
        nn_input = np.array([carstate.speed_x, carstate.distance_from_center, carstate.angle]+list(carstate.distances_from_edge)[0:-1])
        i=0
        while(i <= 20):
            nn_input[i] = (nn_input[i] - Ndata.minarray[i])/(Ndata.maxarray[i]-Ndata.minarray[i])
            i += 1
            
        nn_input.shape = (nn_input.shape[0], 1, nn_input.shape[1])

        with tf.Session() as sess:
            saver1.restore(sess,'./model_accbrk/model_accbrk')
            accbrk = sess.run("prediction:0", feed_dict={"x:0": nn_input})
            #    output = sess.run("prediction", feed_dict={"x:0": tf.cast(input, tf.float32)})
            
            saver2.restore(sess,'./model_steer/model_steer')
            steer = sess.run("prediction:0", feed_dict={"x:0": nn_input})
        #    output = sess.run("prediction", feed_dict={"x:0": tf.cast(input, tf.float32)})
            print("accbrk:", accbrk)
            accbrk = np.round(accbrk)
            print("acc:", accbrk[0])
            print("brk:", accbrk[1])
            print("steer:", steer)

        # GEAR HANDLER

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 3500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        # # manually adjust angle
        # if carstate.angle > 45:
        #     command.accelerator = 0.6
        #     command.steering = 0.5
        # if carstate.angle < -45:
        #     command.accelerator = 0.6
        #     command.steering = -0.5

        # OFFTRACK HANDLER
        # reduce acceleration if offtrack
        acceleration = command.accelerator
        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1 and carstate.distances_from_edge[0] == -1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)
            command.accelerator = min(acceleration, 1)

        # the car is offtrack on the right
        if carstate.distance_from_center < -0.5 and carstate.distances_from_edge[0] == -1:
            if carstate.angle >= -30 and carstate.angle <= -15:
                command.steering = 0
            elif carstate.angle > -15 and carstate.angle <= 120:
                # steer left
                command.steering = 0.8
            elif carstate.angle > 120 or carstate.angle < -30:
                # steer right
                command.steering = -0.8

        # the car is offtrack on the left
        if carstate.distance_from_center > 1.5 and carstate.distances_from_edge[0] == -1:
            if carstate.angle >= 15 and carstate.angle <= 30:
                command.steering = 0
            elif carstate.angle >= -120 and carstate.angle < 15:
                # steer right
                command.steering = -0.8
            elif carstate.angle > 30 or carstate.angle < -120:
                # steer left
                command.steering = 0.8

        # STUCK CAR HANDLER
        if (nn_input[0] < 0.001 and nn_input[0] > -0.001 and command.accelerator > 0.05 and command.gear != -1 and not self.stuck):
            self.stuck_step += 1
            if self.stuck_step > self.stuck_period:
                self.stuck = True
                print ('Stuck')
        else:
            self.stuck_step = 0
        if self.stuck:
            command.gear = -1
            command.steering = -command.steering
            self.stuck_counter += 1
            if self.stuck_counter == self.stuck_recovery:
                self.stuck = False
                self.stuck_counter = 0
                command.gear = 1

        self.drive_step += 1
        return command


