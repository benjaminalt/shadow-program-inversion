# Robot Program Parameter Inference via Differentiable Shadow Program Inversion

This repository contains the source code for the paper

B. Alt, D. Katic, A. K. Bozcuoglu, R. Jäkel, and M. Beetz, “Robot Program Parameter Inference via Differentiable Shadow Program Inversion,” *2021 IEEE International Conference on Robotics and Automation (ICRA)*, Xi’an, China, 2021.

You can find an abstract, the full-text PDF and video material on the paper [website](https://benjaminalt.github.io/spi), 
and an overview of the method in the [blog post](https://benjaminalt.github.io/blog/2021/03/06/shadow-program-inversion.html).

`shadow_program_inversion/shadow_program.py` and `shadow_program_inversion/shadow_skill.py` contain the core implementation
of SPI. `shadow_program_inversion/experiments/contact` contains code to reproduce experiment B.2) of the paper.

### Installation

The following installation instructions have been tested with Python 3.6 on Ubuntu 18.04.

1. Clone the repository:

```
git clone git@github.com:benjaminalt/shadow-program-inversion.git
cd shadow-program-inversion
```

2. *(Optional)* SPI requires PyTorch and several other scientific computing and machine learning libraries. 
   To keep dependencies organized, we suggest creating a virtual environment:

```
python3 -m pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

3. Install all dependencies, and optionally install shadow-program-inversion into your virtual environment 
   for easier access.

```
pip install -f requirements.txt
python setup.py install             # Optional
```

4. *(Optional)* [Set up your robot](#robot-setup) to reproduce the real-world experiments. 
   If you use our provided dataset and pre-trained models, this is not required.
   
### Download dataset and trained models

We provide a dataset containing simulated and real robot data for contact motions on foam, rubber and PCB surfaces,
as well as trained Shadow Skills for both DMPs and URScript programs. To download the dataset and trained models, 
`cd` into the repository root and execute `python download_data.py`. This will download an archive of ~700MB, which will
be decompressed to ~18GB. The dataset can also be found [here]("https://seafile.zfn.uni-bremen.de/f/7e588dd285de4d7486a6/?dl=1").

After the download, your repository should look like this:
```
shadow-program-inversion
|--data
|--optimized_parameters
|--results
|--shadow_program_inversion
|--trained_models
```

### Reproduce our results

`shadow_program_inversion/experiments/contact` contains code to reproduce experiment B.2) in the paper 
("Data-driven optimization of contact forces"/"Generalization  to  different  skill  representations")
for DMP and URScript program representations.

To reproduce the full experiment, comprising generation of simulated data, collection of real data, training of
an autoregressive neural prior, training of the actual shadow skill, parameter inference and evaluation of the resulting
you can run the following scripts in turn (here for optimizing DMP parameters for contact with a PCB):
```
cd shadow_program_inversion/experiments/contact/dmp
python generate_sim_data.py pcb 100000                      # Generate 100000 simulated contact motions
python train_prior.py pcb                                   # Train an autoregressive prior on the simulated data
python collect_real_data.py pcb 1000                        # Execute 1000 contact motions on the real robot
python train_shadow_skill.py pcb                            # Train the Shadow Skill on the real data
python optimize_parameters.py pcb 5N --n=10 --show_plots    # Infer 10 sets of optimal program parameters for a target force of 5N
python test_optimized_parameters.py pcb 5N --show_plots     # Test the resulting program parameters on a real robot
cd ..
python analyze_results.py --plot-series=dmp_pcb_5N          # Compute statistics and display results
```

**If you downloaded the provided dataset and trained models, you can skip any of the above steps** and skip directly
to the parameter optimization (`optimize_parameters.py`) or analysis (`analyze_results.py`).

### Robot setup

If you want to collect your own data and apply our method directly on your robot, you will need to
[install ROS](http://wiki.ros.org/melodic/Installation/Ubuntu) (tested with Melodic) and the
[Universal Robots ROS driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver). Make sure to correctly
[set up the ROS PC and robot](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver#setting-up-a-ur-robot-for-ur_robot_driver).
Finally, you need to set `ROBOT_IP` in `shadow_program_inversion/utils/config.py` to the IP address of your robot.
The scripts which communicate directly with the robot (`collect_real_data.py` and `test_optimized_parameters.py`)
require ROS to be running and the [remote control program](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/blob/master/ur_robot_driver/doc/install_urcap_e_series.md) to be running on the robot.
For a UR5e, you can start ROS by running `roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=172.16.53.128`
(make sure you substitute your robot's IP) and ROS will automatically try to connect to the remote control program
on the robot.