from Grid_World_Environment_rewards2 import GridWorldEnvironment_rewards2
from Double_Dueling_DQN import Double_Dueling_DQN as DDDQN
from Uniform_Experience_Replay_Buffer import Uniform_Experience_Replay_Buffer as UERB
import pickle
import tensorflow as tf
import numpy as np
import time
import copy

load_stuff=False
environment_name="Grid_World_rewards2"
###############################################################################################
#Helper Functions:
###############################################################################################
def get_policy(start_states):
    policy = {}
    for st in start_states:
        dddqn.Environment.set_state(copy.copy(st))
        policy[tuple(st)]=int(dddqn.session.run(dddqn.online_network.forward_action_graph,
                                                feed_dict={dddqn.online_network.X_in: np.reshape(dddqn.Environment.encode_state(), (1, input_size))}))
    return policy

def print_policy(dddqn, policy):
    for row in range(dddqn.Environment.num_rows):
        print("|", end="")
        for column in range(dddqn.Environment.num_columns):
            if (row, column) in policy:
                print("{0:5}".format(dddqn.Environment.action_map[policy[(row, column)]]), end="")
            else:
                print("{0:5.2f}".format(dddqn.Environment.rewards_map[row][column]), end="")
            print("|", end="")
        print()
    return

def print_state_values(dddqn, start_states, network="Online"):
    if network=="Online":
        initial_start_state=dddqn.Environment.state
        for row in range(dddqn.Environment.num_rows):
            print("|", end="")
            for column in range(dddqn.Environment.num_columns):
                dddqn.Environment.set_state([row, column])
                if [row, column] in start_states:
                    print("{0:5.2f}|".format(max(dddqn.session.run(dddqn.online_network.forward_values_graph,
                                                                feed_dict={dddqn.online_network.X_in: np.reshape(dddqn.Environment.encode_state(), (1, input_size))})[0])), end="")
                else:
                    print("{0:5.2f}|".format(dddqn.Environment.rewards_map[row][column]), end="")
            print()
            dddqn.Environment.set_state(initial_start_state)
        return
    elif network=="Target":
        initial_start_state=dddqn.Environment.state
        for row in range(dddqn.Environment.num_rows):
            print("|", end="")
            for column in range(dddqn.Environment.num_columns):
                dddqn.Environment.set_state([row, column])
                if [row, column] in start_states:
                    print("{0:5.2f}|".format(max(dddqn.session.run(dddqn.target_network.forward_values_graph,
                                                                feed_dict={dddqn.target_network.X_in: np.reshape(dddqn.Environment.encode_state(), (1, input_size))})[0])), end="")
                else:
                    print("{0:5.2f}|".format(dddqn.Environment.rewards_map[row][column]), end="")
            print()
            dddqn.Environment.set_state(initial_start_state)
            return
###############################################################################################
#Disable AVX and AVX2 warnings
###############################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###############################################################################################
#Instantiate Replay Buffer
###############################################################################################
line_width=100
print("="*line_width)
print("Pre-Training")
print("="*line_width)
print("Instantiating Replay Buffer...")
buffer_capacity=100000
batch_size=64
replay_buffer=UERB(capacity=buffer_capacity, batch_size=batch_size)
load_buffer=load_stuff
buffer_file_name='replay_buffer'
if load_buffer:
    print("Loading buffer...")    
    infile=open(buffer_file_name,'rb')
    loaded_deque=pickle.load(infile)
    infile.close()
    replay_buffer.buffer=loaded_deque    
###############################################################################################
#Instantiate Environment
###############################################################################################
print("Instantiating " + environment_name + " Environment...")
game_map=np.array([[-1, 1, 1, 1, 1, 1]]) #1: non-terminal, -1: terminal, 0: obstacle
rewards_map=np.array([[-1, -2, -3, -2, -1, 0]])
penalty=0.10
Environment=GridWorldEnvironment_rewards2(num_actions=4, game_map=game_map, rewards_map=rewards_map, penalty=penalty)
print("Environment Map:")
Environment.print_map()
###############################################################################################
#Instantiate Double Dueling DQN
###############################################################################################
print("Instantiating Double Dueling DQN...")
input_size=len(Environment.encode_state())
output_size=4 #should be the number of actions
architectures = {"Hidden": (128, 128),
                 "Value": (64, 1),
                 "Advantage": (64, output_size)}
transfer_rate=2000
gamma=0.99
learning_rate=0.00000001
sess = tf.Session()
dddqn = DDDQN(input_size=input_size,
              output_size=output_size,
              architextures=architectures,
              transfer_rate=transfer_rate,
              gamma=gamma,
              learning_rate=learning_rate,
              Environment=Environment,
              replay_buffer=replay_buffer,
              session=sess)
##########################################################################################
#Restore model
##########################################################################################
model_path=environment_name+"_model.ckpt"
saver=tf.train.Saver()
restore=load_stuff
if restore:
	print("Restoring Weights...")
	saver.restore(sess, "./"+model_path)
	dddqn.copy_weights() #VERY IMPORTANT SINCE TARGET NETWORK IS NOT SAVED
else:
	print("Initializing Network Weights...")
	sess.run(tf.global_variables_initializer())
##########################################################################################
#Restore/Declare lists for matplotlib
##########################################################################################
#lists to store values for graphs
store_rate=100 #how often to store values
load_arrays=load_stuff
losses_file_name=environment_name+"_losses"
lr_file_name=environment_name+"_learning_rates"
eps_file_name=environment_name+"_epsilons"
if not load_arrays:
    losses=[]
    learning_rates=[]
    epsilons=[]
else:
    print("Loading lists for matplotlib...")
    losses=list(np.load(losses_file_name))
    learning_rates=list(np.load(lr_file_name))
    epsilons=list(np.load(eps_file_name))
#########################################################################################
#Fill buffer
#########################################################################################
if not load_buffer:
    print("Filling Buffer...")
    dddqn.initialize_replay_buffer()
#########################################################################################
#get pre-training policy
#########################################################################################
start_states=dddqn.Environment.get_start_states()
print("Getting pre-training policy...")
pre_policy=get_policy(start_states)
#########################################################################################
#print pre-training policy
#########################################################################################
print("Pre-training policy:")
print_policy(dddqn, pre_policy)
#########################################################################################
#print pre-training state values
#########################################################################################
print("Pre-training values:")
print_state_values(dddqn, start_states)
#########################################################################################
#Training
#########################################################################################
print("="*line_width)
print("Training")
print("="*line_width)
tick=time.time()
state=dddqn.Environment.encode_state()
num_epochs=2000
moves_per_epoch=4

#epsilon parameters to linearly decrease epsilon from start_epsilon to final_epsilon over
#num_decrease_steps.
start_epsilon=1
final_epsilon=0.1
num_decrease_steps=1000
epsilon_change=(final_epsilon-start_epsilon)/num_decrease_steps 
if not load_stuff:
	dddqn.epsilon=start_epsilon
else:
	dddqn.epsilon=final_epsilon

for i in range(num_epochs):
    for j in range(moves_per_epoch):
        action=dddqn.epsilon_greedy_action(state)
        dddqn.Environment.print_map()
        print()
        reward, next_state, terminal = dddqn.Environment.take_action(action)
        dddqn.replay_buffer.add((state, action, reward, next_state, terminal))
        if terminal:
            state=dddqn.Environment.initialize_state()
        else:
            state=next_state
    states, actions, rewards, next_states, terminals=dddqn.replay_buffer.get_batch()
    dddqn.train_step(states, actions, rewards, next_states, terminals)
    if i % store_rate==0:
        loss=loss=dddqn.session.run(dddqn.loss, feed_dict={dddqn.online_network.X_in: states,
                                                           dddqn.online_network.y_in: dddqn.get_targets(states, actions, rewards, next_states, terminals, dddqn.session)})
        losses.append(loss)
        learning_rates.append(dddqn.session.run(dddqn.learning_rate))
        epsilons.append(dddqn.epsilon)
    if i < num_decrease_steps and not load_stuff:
        dddqn.epsilon=dddqn.epsilon+epsilon_change
    if i%dddqn.transfer_rate==0 and i>0:
        print("Epoch {} out of {}".format(i, num_epochs))
        print("Copying Weights...")
        dddqn.copy_weights()
        loss=dddqn.session.run(dddqn.loss, feed_dict={dddqn.online_network.X_in: states,
                                                      dddqn.online_network.y_in: dddqn.get_targets(states, actions, rewards, next_states, terminals, dddqn.session)})
        print("Loss: {}".format(loss))
        print("Online network values:")
        print_state_values(dddqn, start_states)
        print("Current Policy:")
        policy=get_policy(start_states)
        print_policy(dddqn, policy)
tock=time.time()
print(tock-tick)
print("="*line_width)
print("Post-Training")
print("="*line_width)
#########################################################################################
#Save model, pickle replay buffer, pickle losses, learning_rates, and epsilons
#########################################################################################
save_path=saver.save(sess, "./"+model_path)
print("="*line_width)
print("Model saved in file: {}".format(save_path))
#pickle replay buffer
outfile=open(buffer_file_name,'wb')
pickle.dump(dddqn.replay_buffer.buffer, outfile)
outfile.close()
print("Buffer saved in file: {}".format(buffer_file_name))
#save losses
losses=np.array(losses)
outfile=open(losses_file_name,'wb')
np.save(outfile, losses)
outfile.close()
print("Losses saved in file: {}".format(losses_file_name))
#save learning_rates
learning_rates=np.array(learning_rates)
outfile=open(lr_file_name,'wb')
np.save(outfile, learning_rates)
outfile.close()
print("Learning rates saved in file: {}".format(lr_file_name))
#save learning_rates
epsilons=np.array(epsilons)
outfile=open(eps_file_name,'wb')
np.save(outfile, epsilons)
outfile.close()
print("Epsilons saved in file: {}".format(eps_file_name))
#########################################################################################
#get post-training policy
#########################################################################################
print("="*line_width)
print("Getting post-training policy...")
post_policy = get_policy(start_states)
#########################################################################################
#print post-training policy
#########################################################################################
print("Post-training policy:")
print_policy(dddqn, post_policy)
#########################################################################################
#Print post-training state values
#########################################################################################
print("Post-training values:")
print_state_values(dddqn, start_states)