# controls_challenge

Hi! This is my solution to the comma.ai Controls Challenge. The original README is below this explanation.

Final Score: **43.776**

I had basically zero prior experience in evolutionary algorithms, control theory, MPC, or behavioral cloning which meant I learned a ton, but if you find yourself saying, “this is weird. why did he do it this way?”, it’s because I didn’t know there was a better way to do it.

All the expensive work for this was done on Middlebury College’s (pretty small) HPC cluster.

I converted the ONNX model to Torch which let me run parallelized physics inference on a GPU. I think this is probably slightly less performant than using the ONNX GPU runtime but it made the code a lot cleaner.

The final solution is broken into 3 parts:

### 1. The Prior: A Small MLP Evolved with CMA-ES
My first attempt was a ~600 parameter MLP directly evolved with CMA-ES. The network takes in 14 derived features including PID, past lat accels, and summary statistics of the future and outputs a single action for this time step.

I played around with having it output a delta on a baseline PID controller but it ended up being worse.

This scored ~55 on the full 5k eval which isn’t competitive but was good enough to serve as a prior for the MPC in the next part.

### 2. The Teacher: Expensive, Parallelized GPU Trajectory Optimization
My next idea was to use the CMA-ES model as a baseline, but at each timestep try many slight variations on its proposed actions. I later learned that this is an established technique (a mix of ideas from MPC), but I didn’t know that when I started implementing it so in the code it's called policy-guided trajectory optimization or PGTO.

At each timestep, we generate 128 candidate action trajectories. We take the CMA-ES prior suggested action and apply random noise to get 128 different actions, we then simulate a step forward with each action and repeat this 6 time steps into the future. We select the top 20% of trajectories (the elite fraction), adjust the noise distribution to be centered around them and iterate until the candidates converge. At that point we actually act on the best initial action found.

The physics model provided returns a probability distribution over the lat accel tokens. Actual rollouts in tinyphysics sample this distribution (with temperature 0.8) to step forward. Initially I matched this behavior in PGTO when rolling forward candidate action trajectories, but very often an action sequence that is on average worse than another will  win out because it got a lucky rollout. The largest gain in the quality of PGTO output came from switching internal rollouts to the expected output of the physics model. This makes candidates directly comparable. Tiny changes in an action sequence have a tiny effect on the rollout rather than leading to a totally different stochastic outcome. The cost of a rollout of expected lat accels is not the same as the expected cost of a rollout of stochastic lat accels (which is the true metric we want to optimize), but it’s close enough in this case to work well.

This whole process ends up being absurdly computationally expensive. The eval itself does 500 physics steps for each of the 5 thousand segments, so 2.5 million steps total. This takes about 10 minutes on my MacBook for the PID controller. This trajectory optimization does ~1k rollouts of 6 timesteps for every single real timestep forward. That’s 15 billion physics steps for 5k segments, so 41 days on my laptop. The control is expected to make a decision every tenth of a second and it takes us 1.5 seconds to make every decision. Not sure the math works out… I was hoping I could find some clever way to optimize this process and use it as an online controller, but it ended up just taking too long to get good results. Here, I came up with the final plan to instead generate these actions offline and use them to distill a student model.

I did a final run, generating 10 different rollouts for all 5k segments. This took 3 days on 8 RTX workstation GPUs and the expected score—if this was used as a true controller—was 43.2. I now had 25 million really high quality state-action pairs (500 timesteps x 5k segments x 10 rollouts) to train the student on.

### 3. The Student: Behavioral Cloning
This final part ended up being easier than expected. It’s pretty much vanilla behavioral cloning with just a couple of additions.

Training used all 25 million samples with MSE loss. I considered filtering out the worst scoring rollouts of each segment, but I found they were actually unlucky rollouts and not a failure of the trajectory optimization so they stayed in.

My first attempt was a 3 million parameter vanilla MLP. The inputs are the current state, the past 20 actions and lat accels, the next 50 future states, the current relative position within a segment, and the validity of the past action buffer (when the control starts it’s all zeros). This worked, but costs were in the 60s due to distribution shift. I thought I was going to have to solve this with DAgger (requiring more expensive optimization), but it was pretty much eliminated with two small changes.

The first was creating separate encoders for the past, future, and current state. These all feed into a shared head. Past actions suffer from distribution shift, so they’re not very reliable. We suggest this to the model by making the future encoder (~700k params) much larger than the past one (~50k params). This all can be represented in a vanilla MLP but the inductive bias was significantly better this way.

The second change was adding noise to past action features during training. This also discourages the model from placing a lot of weight into past actions. Constant gaussian noise with a standard deviation of 0.012 worked exceptionally well to fix distribution shift. Noise annealing from 0.023 to 0.002 worked even better.

Final training was 400 epochs with cosine LR annealing (3e-4 to 1e-6) and noise annealing (0.023 to 0.002). The end model was 1.5 million parameters and trained in 3.5 hours on one RTX 6000 Ada. Final training loss was 0.000026 (which is artificially raised by the 0.002 feature noise).

Model inference for the full eval takes minutes. Much better than 41 days.

---

<div align="center">
<h1>comma Controls Challenge v2</h1>


<h3>
  <a href="https://comma.ai/leaderboard">Leaderboard</a>
  <span> · </span>
  <a href="https://comma.ai/jobs">comma.ai/jobs</a>
  <span> · </span>
  <a href="https://discord.comma.ai">Discord</a>
  <span> · </span>
  <a href="https://x.com/comma_ai">X</a>
</h3>

</div>

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

## Getting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual car and road states from [openpilot](https://github.com/commaai/openpilot) users.

```
# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid
```

There are some other scripts to help you get aggregate metrics:
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

python -m offline.run -n 1 -w 0 -m 100 -d cuda
```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. Its inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`), and a steer input (`steer_action`), then it predicts the resultant lateral acceleration of the car.

## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.

## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(\mathrm{actual{\textunderscore}lat{\textunderscore}accel} - \mathrm{target{\textunderscore}lat{\textunderscore}accel})^2}{\text{steps}} * 100$
- `jerk_cost`: $\dfrac{(\Sigma( \mathrm{actual{\textunderscore}lat{\textunderscore}accel_t} - \mathrm{actual{\textunderscore}lat{\textunderscore}accel_{t-1}}) / \Delta \mathrm{t} )^{2}}{\text{steps} - 1} * 100$

It is important to minimize both costs. `total_cost`: $(\mathrm{lat{\textunderscore}accel{\textunderscore}cost} * 50) + \mathrm{jerk{\textunderscore}cost}$

## Submission
Run the following command, then submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

Competitive scores (`total_cost<100`) will be added to the leaderboard

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller bc --baseline_controller pid
python eval.py --model_path ./models/tinyphysics.onnx --data_path ../images_1/data/test --num_segs 1 --test_controller bc --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma

Like this sort of stuff? You might want to work at comma!
[comma.ai/jobs](https://comma.ai/jobs)
