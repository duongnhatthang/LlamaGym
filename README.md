# Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM

This repository is the official implementation of [Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM](placeholder). 

![](https://github.com/duongnhatthang/LlamaGym/blob/main/figs/loro.png)
## Requirements

To install requirements:


```setup
pip install -r requirements.txt
pip install git+https://github.com/mila-iqia/atari-representation-learning.git
d3rlpy install d4rl
```
Install Mujoco: https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da


## Training

To collect the data from LLM, choose the model name and environment by editing the 'hyperparams' variable in llm_main.py. Then, run this command:

```llm_data_collection
python llm_main.py
```

To pretrain the model(s) in the paper, choose the number of pretrain episode and number of pretrain steps by editing the 'hyperparams' variable (n_pretrain_eps, n_pretrain_steps) in pretrain_from_llm.py. Changing the file paths from the previous step in the 'get_llm_data_paths' funciton in pretrain_from_llm.py. Then, run this command:

```pretrain
python pretrain_from_llm.py
```

To fine-tune the RL algorithm on top of the pretrain models, after completing the previous two steps, choose the environment, the number of pretrain and online episodes by editing the 'hyperparams' variable in online_main.py and run this command:

```fine_tune
python online_main.py
```

You can follow the same step above to collect on-policy (pure RL) data by running the on_policy_pretrain_exp.py file.

## Evaluation

To visualize the results, use the visualization.ipynb notebook. You can run this directly to visualize the results shown in the paper.

## Results

Our model achieves the following performance on six OpenAI Gym environments:

![](https://github.com/duongnhatthang/LlamaGym/blob/main/figs/main_results.png)

## [Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM](Placeholder):
We investigate the usage of Large Language Model (LLM) in collecting high-quality data to warm-start Reinforcement Learning (RL) algorithms for learning in some classical Markov Decision Process (MDP) environments. In this work, we focus on using LLM to generate an off-policy dataset that sufficiently covers state-actions visited by optimal policies, then later using an RL algorithm to explore the environment and improve the policy suggested by the LLM. Our algorithm, LORO, can both converge to an optimal policy and have a high sample efficiency thanks to the LLM's good starting policy. On multiple OpenAI Gym environments, such as CartPole and Pendulum, we empirically demonstrate that LORO outperforms baseline algorithms such as pure LLM-based policies, pure RL, and a naive combination of the two, achieving up to four times the cumulative rewards of the pure RL baseline.

## Contributing

[Apache 2.0](https://github.com/duongnhatthang/LlamaGym/blob/main/LICENSE)

## Note

The code is referencing this [repo](https://github.com/KhoomeiK/LlamaGym). The environment descriptions are referenced from this [repo](https://github.com/mail-ecnu/Text-Gym-Agents).

## Citation
Placeholder
