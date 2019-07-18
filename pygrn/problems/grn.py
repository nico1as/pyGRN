from .base import Problem
import numpy as np
import gym
from gym_jsbsim.catalogs.catalog import Catalog as prp
import random
import math


class GymGRN(Problem):

    def __init__(self, env_name, nin, nout, ep_max):
        self.nin = nin
        self.nout = nout
        self.env = gym.make(env_name)
        self.ep_max = ep_max
        
    def eval(self, grn, de=False, deo=False):
        #print("eval::init fitness")
        fitnesses = np.zeros(self.ep_max)
        #print("eval::init grn")
        self.grn_init(grn)
        fitness = 0.0
        #for e in range(self.ep_max):
        end = False
        fit = 0
        states = self.env.reset()
        step = 0
        #print("eval::start simu")
        action_state_list = ""
        simple_state_list = ""
        #grn.warmup(50)
        debug = de
        debug_out = deo
        while not end:
            norm_states = []
            for i in range(self.nin):
                norm_states += [(states[i]-self.env.observation_space[i].low[0])/(self.env.observation_space[i].high[0] - self.env.observation_space[i].low[0])]
            if debug:
                print("----- time (s): ", self.env.sim.get_property_value(prp.simulation_sim_time_sec), " -----")
                print("\t STATES")
                print("\t\t Simu states: ", states)
                print("\t\t Normalize states: ", norm_states)
                print("\t\t Velocity: ", self.env.sim.get_property_value(prp.velocities_vc_fps))
                print("\t\t dist: ", self.env.sim.get_property_value(prp.shortest_dist))
            grn.set_input(norm_states)
            grn.step()
            actions = grn.get_output()
            real_actions = []
            for a in range(int(len(actions)/2)):
                if actions[a*2]+actions[a*2+1] != 0:
                    if self.env.action_space[a].low[0]==-1:
                        # value between -1 and 1: (s1-s2) / (s1+s2)
                        real_actions += [(actions[a*2]-actions[a*2+1])/(actions[a*2]+actions[a*2+1])]
                    else:
                        # value between 0 and 1: s1 / (s1+s2) or |(s1-s2)| / (s1+s2)
                        real_actions += [actions[a*2]/(actions[a*2]+actions[a*2+1])]
                else:
                    real_actions += [0]
            
            if debug:
                print("\t ACTIONS")
                print("\t\t GRN Actions: ", actions)
                print("\t\t Normalize actions: ", real_actions)
            states, reward, end, _ = self.env.step(real_actions)
            
            if self.env.sim.get_property_value(prp.simulation_sim_time_sec) < 5.0:
                fit += 0.0
            else:
                fit += reward * reward
            step += 1

            if debug:
                print("\t REWARD")
                print("\t\t Simu reward: ", reward)
                print("\t\t GRN Fitness: ", fit)

            lon = self.env.sim.get_property_value(prp.position_long_gc_deg)
            lat = self.env.sim.get_property_value(prp.position_lat_geod_deg)
            alt = self.env.sim.get_property_value(prp.position_h_sl_ft)
            sim_time = self.env.sim.get_property_value(prp.simulation_sim_time_sec)
            sim_freq = self.env.sim.get_property_value(prp.simulation_dt)
            
            simple_state_list = simple_state_list + "\n" + str(sim_time) + ", " + str(lon) + ", " + str(lat) + "," + str(alt) + ", "
            action_state_list = action_state_list + "\n" +  "," + str(real_actions[0]) + "," + str(real_actions[1]) + "," + str(real_actions[2])+ "," + str(fit) #str(real_actions[3]) + ", " + str(fit)
            #fitnesses[e] = fit
        #np.sort(fitnesses)
        #fit = 0
        #sum_e = 0
        #for e in range(self.ep_max):
        #    fit += fitnesses[e] * (e + 1)
        #    sum_e += e + 1
        #print("eval::compute fitness: ", fit)

        ### PRINT
        if debug_out:
            print("\n")
            print("---------------- AIRCRAFT POINTS: LON, LAT, ALT ----------------")
            print("\n")
            print(simple_state_list)
            print("\n")

        #print("\n")
        #print("---------------- Action ----------------")
        #print("\n")
        #print(action_state_list)
        #print("\n")

        return fit#/ sum_e
