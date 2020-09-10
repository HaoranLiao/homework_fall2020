# command to run Question 1 part 2 Ant task BC:
--expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name q1_bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --train_batch_size 1000 --num_agent_train_steps_per_iter 1000

# command to run Question 1 part 2 HalfCheeta task BC:
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name q1_bc_halfcheeta --n_iter 1 --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --train_batch_size 1000 --num_agent_train_steps_per_iter 100

# command to run Question 1 part 3 Ant task with varying hyperparameter num_agent_train_steps_per_iter BC:
--expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name q1_bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --train_batch_size 1000 
# the following flag was varied with the following values each for one run
--num_agent_train_steps_per_iter [100, 500, 1000, 2500, 5000]

# command to run Question 2 Ant task DAgger:
--expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name q2_dagger_ant --n_iter 10 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --video_log_freq -1 --do_dagger

# command to run Question 2 HaflCheeta task DAgger:
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl --env_name HalfCheetah-v2 --exp_name q1_dagger_halfcheetah --n_iter 10 --expert_data cs285/expert_data/expert_data_HalfCheetah-v2.pkl --video_log_freq -1 --do_dagger
