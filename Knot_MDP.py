from Double_Dueling_DQN import Double_Dueling_DQN as DDDQN
from Uniform_Experience_Replay_Buffer import Uniform_Experience_Replay_Buffer as UERB
from Slice_Environment_Wrapper import SliceEnvironmentWrapper as SEW
from Start_States_Buffer import Start_States_Buffer as SSB
import pickle
import tensorflow as tf
import numpy as np
import time
import copy

load_stuff=False
###############################################################################################
#Disable AVX and AVX2 warnings
###############################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
###############################################################################################
#Instantiate Replay Buffer
###############################################################################################
line_width=100
print("="*line_width)
print("Pre-Training")
print("="*line_width)
print("Instantiating Replay Buffer...")
buffer_capacity=10000
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
#Instantiate Start States Buffer
###############################################################################################    
start_states_capacity=1000
max_braid_index=3
max_braid_length=5
seed_braids=[[1, 1, 1]]
starts_buffer=SSB(capacity=start_states_capacity,
                  max_braid_index=max_braid_index,
                  max_braid_length=max_braid_length,
                  seed_braids=seed_braids)
###############################################################################################
#Instantiate Environment
###############################################################################################
environment_name="SliceEnv"
print("Instantiating " + environment_name + " Environment...")
#max_braid_index=3 
#max_braid_length=5
inaction_penalty=0.05
Environment=SEW(max_braid_index=max_braid_index,
                max_braid_length=max_braid_length,
                inaction_penalty=inaction_penalty,
                start_states_buffer=starts_buffer)
###############################################################################################
#Instantiate Double Dueling DQN
###############################################################################################
print("Instantiating Double Dueling DQN...")
input_size=len(Environment.slice.encode_state())
output_size=13 #should be the number of actions
architectures = {"Hidden": (256, 256, 256),
                 "Value": (128, 1),
                 "Advantage": (128, output_size)}
transfer_rate=2000
gamma=0.99
learning_rate=0.0000001
sess=tf.Session()
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
#Restore model or Initialize Weights
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
losses_file_name="losses"
lr_file_name="learning_rates"
eps_file_name="epsilons"
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
    dddqn.initialize_replay_buffer(display=False)
#########################################################################################
#Training
#########################################################################################
print("="*line_width)
print("Training")
print("="*line_width)
tick=time.time()
state=dddqn.Environment.slice.encode_state()
num_epochs=20000
moves_per_epoch=4

#epsilon parameters to linearly decrease epsilon from start_epsilon to final_epsilon over
#num_decrease_epochs.
start_epsilon=1
final_epsilon=0.1
num_decrease_epochs=6000
epsilon_change=(final_epsilon-start_epsilon)/num_decrease_epochs 
if not load_stuff:
	dddqn.epsilon=start_epsilon
else:
	dddqn.epsilon=final_epsilon

for i in range(num_epochs):
    for j in range(moves_per_epoch):
        action=dddqn.epsilon_greedy_action(state)
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
    if i < num_decrease_epochs and not load_stuff:
        dddqn.epsilon=dddqn.epsilon+epsilon_change
    if i%dddqn.transfer_rate==0 and i>0:
        print("Epoch {} out of {}".format(i, num_epochs))
        print("Copying Weights...")
        dddqn.copy_weights()
        loss=dddqn.session.run(dddqn.loss, feed_dict={dddqn.online_network.X_in: states,
                                                      dddqn.online_network.y_in: dddqn.get_targets(states, actions, rewards, next_states, terminals, dddqn.session)})
        print("Loss: {}".format(loss))
tock=time.time()
print(tock-tick)
print("="*line_width)
print("Post-Training")
print("="*line_width)
#########################################################################################
#Save model, pickle replay buffer, save losses, learning_rates, and epsilons
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