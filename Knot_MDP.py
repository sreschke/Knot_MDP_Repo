from Double_Dueling_DQN import Double_Dueling_DQN as DDDQN
from Prioritized_Experience_Replay_Buffer import Prioritized_Experience_Replay_Buffer as PERB
from Slice_Environment_Wrapper import Slice_Environment_Wrapper as SEW
from Start_States_Buffer2 import Start_States_Buffer as SSB
from SliceEnvironment import SliceEnv as SE
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import copy
import sys
import os.path


if __name__ == "__main__":
    ###############################################################################################
    #Hyperpararameters
    ###############################################################################################
    if len(sys.argv)==1: #manually define hyperparameters
        print("Manually setting hyperparameters...")
        load_stuff=False #Controls whether the program should load the network weights, replay_buffer, matplotlib lists, etc. from a previous training run
        #name files for matplotlib lists, replay_buffer, model weights etc.
        load_job_name="SliceEnv_try_29" #name used to load files from previous job
        save_job_name="SliceEnv_try_29.1" #name used to save files
        #Replay buffer
        replay_capacity=1000 #needs to be large enough to hold a representative sample of the state space
        batch_size=512
        alpha=0.6 #see section B.2.2 (pg. 14 table 3) in paper: https://arxiv.org/pdf/1511.05952.pdf
        replay_epslion=0.01 #introduced on page 4 in paper: https://arxiv.org/pdf/1511.05952.pdf
        #FIXME: figure out how beta needs to anneal
        beta=0.5 #see page 5 of paper: https://arxiv.org/pdf/1511.05952.pdf
        
        #Start States Buffer
        seed_braids=seed_braids=[[1, 1, 1],
                                 [1, 1, 1, 2, -1, 2], 
                                 [1, 1, -2, 1, -2, -2], 
                                 [1, 1, 1, -2, 1, -2]] #The braids we want the algorithm to solve. Info stored in seed_queue

        start_states_capacity=100000
        max_braid_index=6
        max_braid_length=10

        #Slice Environment Wrapper (Environment)
        uniform=True #when picking a random action, actions are sampled uniformly if uniform=True. Otherwise, actions are selected using distribution defined with action_probabilites
        move_penalty=0.09 #penalty incurred for taking any action
        seed_prob=0.5 #probability of picking from seed_frame when initializing state


        #Double Dueling DQN
        output_size=13 #should be the number of actions the agent can take in the MDP
        architectures={'Hidden': (512, 512, 512), 'Value': (512, 1), 'Advantage': (512, 13)}

        transfer_rate=2000 #how often (in epochs) to copy weights from online network to target network
        gamma=0.99
        learning_rate=0.000000001

        #Training
        euler_char_reset=-8 #algorithm will initialize state if any eulerchar falls below euler_char_reset
        max_actions_length=40 #initialize_state() is called if an episode takes more actions than max_actions_length

        #epsilon parameters to linearly decrease epsilon from start_epsilon to final_epsilon over
        #num_decrease_epochs. If a model is loaded (i.e. load_stuff=True), epsilon wil be the
        #final_epsilon and will not change.
        start_epsilon=1
        final_epsilon=0.1
        num_decrease_epochs=250000
        epsilon_change=(final_epsilon-start_epsilon)/num_decrease_epochs

        store_rate=10000 #how often (in epochs) to store values for matplotlib lists
        report_policy_rate=1000 #how often (in epochs) to report the policies
        num_epochs=2000000 #how many epochs to run the algorithm for
        moves_per_epoch=4
        if not load_stuff:
            assert num_epochs>=num_decrease_epochs, "num_epochs is less than num_decrease_epochs"

    elif len(sys.argv)==2: #load hyperparameters from a dataframe
        assert sys.argv[1].isdigit(), "Got a non-integer command line argument. Got {} which is a {}".format(sys.argv[1], type(sys.argv[1]))
        hyperparameter_file_name="hyperparameter_df"
        assert os.path.exists(hyperparameter_file_name), "The file {} does not exist".format(hyperparameter_file_name)
        row_index=int(sys.argv[1])
        print("Loading hyperparameters from file {} using row_index {}...".format(hyperparameter_file_name, row_index))
        df=pd.read_msgpack(hyperparameter_file_name)
        df=df.iloc[row_index]
        load_stuff=df.loc["load_stuff"]
        load_job_name=df.loc["load_job_name"]
        save_job_name=df.loc["save_job_name"]
        replay_capacity=int(df.loc["replay_capacity"])
        batch_size=int(df.loc["batch_size"])
        seed_braids=seed_braids=[list(x) for x in df.loc["seed_braids"]]
        start_states_capacity=int(df.loc["start_states_capacity"])
        max_braid_index=int(df.loc["max_braid_index"])
        max_braid_length=int(df.loc["max_braid_length"])
        uniform=bool(df.loc["uniform"])
        move_penalty=df.loc["move_penalty"]
        seed_prob=df.loc["seed_prob"]
        output_size=int(df.loc["output_size"])
        architectures=df.loc["architextures"]
        transfer_rate=int(df.loc["transfer_rate"])
        gamma=df.loc["gamma"]
        learning_rate=df.loc["learning_rate"]
        euler_char_reset=int(df.loc["euler_char_reset"])
        max_actions_length=int(df.loc["max_actions_length"])
        start_epsilon=df.loc["start_epsilon"]
        final_epsilon=df.loc["final_epsilon"]
        num_decrease_epochs=int(df.loc["num_decrease_epochs"])
        epsilon_change=(final_epsilon-start_epsilon)/num_decrease_epochs
        store_rate=int(df.loc["store_rate"])
        report_policy_rate=int(df.loc["report_policy_rate"])
        num_epochs=int(df.loc["num_epochs"])
        moves_per_epoch=int(df.loc["moves_per_epoch"])
        if not load_stuff:
            assert num_epochs>=num_decrease_epochs, "num_epochs is less than num_decrease_epochs"

    #construct hyperparameters dict - used to print hyperparameters in .out file
    hyperparameters={"replay_capacity": replay_capacity,
                     "batch_size": batch_size,
                     "alpha": alpha,
                     "replay_epsilon": replay_epslion,
                     "seed_braids": seed_braids,
                     "start_states_capacity": start_states_capacity,
                     "max_braid_index": max_braid_index,
                     "max_braid_length": max_braid_length,
                     "uniform": uniform,
                     "move_penalty": move_penalty,
                     "seed_prob": seed_prob,
                     "output_size": output_size,
                     "architectures": architectures,
                     "transfer_rate": transfer_rate,
                     "gamma": gamma,
                     "learning_rate": learning_rate,
                     "max_actions_length": max_actions_length,
                     "euler_char_reset": euler_char_reset,
                     "start_epsilon": start_epsilon,
                     "final_epsilon": final_epsilon,
                     "num_decrease_epochs": num_decrease_epochs,
                     "store_rate": store_rate,
                     "report_policy_rate": report_policy_rate,
                     "num_epochs": num_epochs,
                     "moves_per_epoch": moves_per_epoch}


    ###############################################################################################
    #Helper functions
    ###############################################################################################
    def get_policy(braid, max_braid_index, max_braid_length, session, max_actions_length):
        """Used to get the current policies of the seed_braids during training. Returns
        an action list and the achieved score"""
        actions=[]
        slice=SE(braid, max_braid_index, max_braid_length)
        while not slice.is_Terminal():
            if len(actions) < max_actions_length:
                action=sess.run(dddqn.online_network.forward_action_graph,
                                feed_dict={dddqn.online_network.X_in: np.reshape(slice.encode_state(), (1, dddqn.online_network.input_size))})[0]
            else:
                action=dddqn.Environment.slice.inverse_action_map["Remove Crossing"]
            slice.action(action)
            actions.append(action)
        return actions, slice.eulerchar[1]


    def print_hyperparameters(hyperparameters):
        print("Hyperparameters:")
        for key, value in hyperparameters.items():
            print("\t{}={}".format(key, value))
    ###############################################################################################
    #Disable AVX and AVX2 warnings
    ###############################################################################################
    tick=time.time()
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    ###############################################################################################
    #Print hyperparameters
    ###############################################################################################
    line_width=100
    print("Starting " + save_job_name + "...")
    if load_stuff:
        print("Will load files from " + load_job_name)
    print_hyperparameters(hyperparameters)
    ###############################################################################################
    #Instantiate Replay Buffer
    ###############################################################################################
    print("="*line_width)
    print("Pre-Training")
    print("="*line_width)
    print("Instantiating Replay Buffer...")
    replay_buffer=PERB(memory_size=replay_capacity, batch_size=batch_size, alpha=alpha, epsilon=replay_epslion)
    load_buffer=load_stuff
    load_buffer_file_name=load_job_name+'_replay_buffer'
    if load_buffer:
        assert True==False, "loading replay buffer has not been implemented"
    ###############################################################################################
    #Instantiate Start States Buffer
    ###############################################################################################    
    load_start_states_file_name=load_job_name+"_start_states"
    starts_buffer=SSB(capacity=start_states_capacity,
                      max_braid_index=max_braid_index,
                      max_braid_length=max_braid_length,
                      seed_braids=seed_braids,
                      move_penalty=move_penalty)
    if load_stuff:
        print("Loading Start States Buffer...")
        infile=open(load_start_states_file_name,'rb')
        loaded_deque=pickle.load(infile)
        infile.close()
        starts_buffer.explore_queue=loaded_deque
    ###############################################################################################
    #Instantiate Environment
    ###############################################################################################
    environment_name="SliceEnv"
    print("Instantiating " + environment_name + " Environment...")
    Environment=SEW(max_braid_index=max_braid_index,
                    max_braid_length=max_braid_length,
                    inaction_penalty=move_penalty,
                    start_states_buffer=starts_buffer,
                    seed_prob=seed_prob,
                    uniform=uniform)
    ###############################################################################################
    #Instantiate Double Dueling DQN
    ###############################################################################################
    print("Instantiating Double Dueling DQN...")
    input_size=len(Environment.slice.encode_state())
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
    load_model_path=load_job_name+"_model.ckpt"
    saver=tf.train.Saver()
    restore=load_stuff
    if restore:
        print("Restoring Weights...")
        saver.restore(sess, "./"+load_model_path)
        dddqn.copy_weights() #VERY IMPORTANT SINCE TARGET NETWORK IS NOT SAVED
    else:
        print("Initializing Network Weights...")
        sess.run(tf.global_variables_initializer())
    ##########################################################################################
    #Restore/Declare lists for matplotlib
    ##########################################################################################
    #lists to store values for graphs
    load_arrays=load_stuff
    load_losses_file_name=load_job_name+"_losses"
    load_lr_file_name=load_job_name+"_learning_rates"
    load_eps_file_name=load_job_name+"_epsilons"
    if load_arrays:
        print("Loading lists for matplotlib...")
        losses=list(np.load(load_losses_file_name))
        learning_rates=list(np.load(load_lr_file_name))
        epsilons=list(np.load(load_eps_file_name))
    else:
        losses=[]
        learning_rates=[]
        epsilons=[]
    ###############################################################################################
    #Get pre-training policies
    ###############################################################################################
    print("Getting pre-training policies...")
    for braid in seed_braids:
        actions, score = get_policy(braid, max_braid_index, max_braid_length, sess, max_actions_length)
        print("\tPolicy for braid {}: {}".format(braid, actions))
        print("\tAchieved Euler characteristic: {}".format(score))
    #########################################################################################
    #Fill replay buffer
    #########################################################################################
    if not load_buffer:
        print("Filling Buffer...")
        dddqn.initialize_replay_buffer(display=False, euler_char_reset=euler_char_reset, max_actions_length=max_actions_length)
    tock=time.time()
    print("Pre-training set-up took {} seconds".format(tock-tick))
    #########################################################################################
    #Training
    #########################################################################################
    print("="*line_width)
    print("Training")
    print("="*line_width)
    tick=time.time()
    state=dddqn.Environment.slice.encode_state()
 
    if not load_stuff:
        dddqn.epsilon=start_epsilon
    else:
        dddqn.epsilon=final_epsilon

    actions_list=[]
    for i in range(num_epochs):
        for j in range(moves_per_epoch):
            if len(actions_list)<max_actions_length and not dddqn.check_eulerchars(euler_char_reset):
                action=dddqn.epsilon_greedy_action(state)
            else:
                action=dddqn.Environment.slice.inverse_action_map["Remove Crossing"] #policy after max_actions_length actions is to remove all crossings
            actions_list.append(action)
            reward, next_state, terminal = dddqn.Environment.take_action(action)
            priority=dddqn.calculate_priorities([state], [action], [reward], [next_state], [terminal])[0]
            dddqn.replay_buffer.add(data=(state, action, reward, next_state, terminal), priority=priority)
            if terminal:
                state=dddqn.Environment.initialize_state()
                actions_list=[]
            else:
                state=next_state
        transitions, weights, indices = dddqn.replay_buffer.get_batch(beta)
        #reshape weights
        #FIXME: figure out how to pass weights into train_step
        weights = np.tile(np.reshape(weights, newshape=(batch_size, 1)), output_size)
        #FIXME figure out how we need anneal beta
        states, actions, rewards, next_states, terminals = zip(*transitions) #unzip transitions as tuples
        priorities=dddqn.calculate_priorities(states, actions, rewards, next_states, terminals)
        dddqn.replay_buffer.priority_update(indices, priorities)
        dddqn.train_step(states, actions, rewards, next_states, terminals, weights)
        if i % store_rate==0:
            loss=dddqn.session.run(dddqn.loss, feed_dict={dddqn.online_network.X_in: states,
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
            if loss > 1000:
                print("The algorithm diverged. Ending run.")
                sys.exit(0)
        if i%report_policy_rate==0 and i>0:
            print("Policies at epoch {}:".format(i))
            for braid in seed_braids:
                actions, score = get_policy(braid, max_braid_index, max_braid_length, sess, max_actions_length)
                print("\tPolicy for braid {}: {}".format(braid, actions))
                print("\tAchieved Euler characteristic: {}".format(score))
    tock=time.time()
    print("Training took {} seconds".format(tock-tick))
    print("="*line_width)
    print("Post-Training")
    print("="*line_width)
    #########################################################################################
    #Save model, pickle replay buffer, save losses, learning_rates, and epsilons, save dataFrame
    #########################################################################################
    save_model_path=save_job_name+"_model.ckpt"
    save_path=saver.save(sess, "./"+save_model_path)
    print("="*line_width)
    print("Model saved in file: {}".format(save_path))
    #pickle replay buffer
    save_buffer_file_name=save_job_name+'_replay_buffer'
    outfile=open(save_buffer_file_name,'wb')
    pickle.dump(dddqn.replay_buffer.buffer, outfile)
    outfile.close()
    print("Replay Buffer saved in file: {}".format(save_buffer_file_name))
    #save losses
    save_losses_file_name=save_job_name+"_losses"
    losses=np.array(losses)
    outfile=open(save_losses_file_name,'wb')
    np.save(outfile, losses)
    outfile.close()
    print("Losses saved in file: {}".format(save_losses_file_name))
    #save learning_rates
    save_lr_file_name=save_job_name+"_learning_rates"
    learning_rates=np.array(learning_rates)
    outfile=open(save_lr_file_name,'wb')
    np.save(outfile, learning_rates)
    outfile.close()
    print("Learning rates saved in file: {}".format(save_lr_file_name))
    #save epsilons
    save_eps_file_name=save_job_name+"_epsilons"
    epsilons=np.array(epsilons)
    outfile=open(save_eps_file_name,'wb')
    np.save(outfile, epsilons)
    outfile.close()
    print("Epsilons saved in file: {}".format(save_eps_file_name))
    #save explore_queue
    save_start_states_file_name=save_job_name+"_start_states"
    outfile=open(save_start_states_file_name,'wb')
    pickle.dump(Environment.start_states_buffer.explore_queue, outfile)
    print("Explore queue saved in file: {}".format(save_start_states_file_name))
    ###############################################################################################
    #Get post-training policies
    ###############################################################################################
    print("Getting post-training policies...")
    for braid in seed_braids:
        actions, score = get_policy(braid, max_braid_index, max_braid_length, sess, max_actions_length)
        print("\tPolicy for braid {}: {}".format(braid, actions))
        print("\tAchieved Euler characteristic: {}".format(score))