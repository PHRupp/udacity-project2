# Project Details

This project trains an agent to interact with Udacity's Reacher where the agent controls an arm that gains +0.1 reward for keeping the arm inside the floating ball. 

The code is written in PyTorch and Python 3.

## Environment
The below is a paraphrasing from the Udacity course's repo regarding this project's environment:

The goal of the agent is to keep the arm within the floating ball as long as possible. The goal is to get an average score of >=30 over 100 consecutive episodes.

According to Udacity: "The state space has 33 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction." 

The agent's action space consists of a 4 dimensional vector where each value is bounded [-1, 1].

## Getting Started

After following the instructions defined here for downloading and installing: https://github.com/udacity/deep-reinforcement-learning/tree/master

My installation was based on 
* Windows 11 x64
* Python 3.6.13 :: Anaconda, Inc.
* Reacher Unity Build: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip

```bash
# Instructions from Deep RL course for environment setup
conda create --name drlnd python=3.6 
activate drlnd

# after creating new conda python environment
cd <path/to/dev/directory>
git clone https://github.com/udacity/deep-reinforcement-learning.git
git clone https://github.com/PHRupp/udacity-project2.git
pushd deep-reinforcement-learning/python

# HACK: edit requirements.txt in deep-rl env to make "torch==0.4.1" since i couldnt get a working version for 0.4.0
pip install https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl
pip install .

# install packages used specifically in my code
pip install matplotlib==3.3.4, numpy==1.19.5
popd
pushd udacity-project2
```

## Usage

In order to run the code, we run it straight via python instead of using jupyter notebooks.

As depicted in the Report.pdf, you can change the paramters in main.py to get different results. Otherwise, you can run the code as-is to get the same results assuming a random seed = 0. 

```python
# run from 'udacity-agent-is-bananas' directory
python main.py
```

