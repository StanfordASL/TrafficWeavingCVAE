# TrafficWeavingCVAE

Code accompanying "[Multimodal Probabilistic Model-Based Planning for Human-Robot Interaction](https://arxiv.org/abs/1710.09483)" (Edward Schmerling, Karen Leung, Wolf Vollprecht, Marco Pavone).

Video: http://stanford.edu/~schmrlng/Schmerling.Leung.Vollprecht.Pavone.ICRA18.mp4

Raw dataset: http://stanford.edu/~schmrlng/human_human_traffic_weaving_rosbags.zip

## Usage
Run `setup.py` located in the base directory of this repository before running anything else and follow the prompts (default choices in parentheses) in order to download/unzip the dataset linked above, as well as symlink the ROS package `traffic_weaving_prediction` and Julia package `TrafficWeavingPlanner` into their correct locations.
- [0. Bag Extraction](0.%20Bag%20Extraction.ipynb)
  - Data preprocessing: runs immediately with the ROS and Python dependencies listed below.
- [1. CVAE Training.ipynb](1.%20CVAE%20Training.ipynb)
  - Model training: runs immediately with the Python dependencies listed below.
- [2. Exploring Predictions.ipynb](2.%20Exploring%20Predictions.ipynb)
  - Visualizing predicted human action sequence distributions: requires an exported model from step 1 and the ROS, Python, and Julia dependencies listed below. Before using this notebook  
  `roslaunch traffic_weaving_prediction predict.launch model:={exported_model_dir}`  
  (wraps [`cvae_node_predict.py`](traffic_weaving_prediction/scripts/cvae_node_predict.py)). Pretrained models (the results of running steps 0 and 1) may be found in the `pretrained_models` subdirectory of this repository, e.g., you may use `{exported_model_dir} = .../TrafficWeavingCVAE/pretrained_models/slim_features_default_hps`.
- [3. Human-in-the-Loop Simulation.ipynb](3.%20Human-in-the-Loop%20Simulation.ipynb)
  - Testing prediction model and policy construction: requires an exported model from step 1 and the ROS, Python, Julia, and VTD dependencies listed below. Before using this notebook  
  `roslaunch vtd_interface human_in_the_loop.launch`  
  (handles VTD simulator input/output) and  
  `roslaunch traffic_weaving_prediction policy.launch model:={exported_model_dir}`  
  (wraps [`cvae_node_scoring.py`](traffic_weaving_prediction/scripts/cvae_node_scoring.py)).
- [4. ICRA18 Videos.ipynb](4.%20ICRA18%20Videos.ipynb)
  - Visualizing predictions over the duration of a traffic weaving interaction: requires ROS, Python, and Julia dependencies below, and  
  `roslaunch traffic_weaving_prediction predict.launch`.

## Dependencies
- ROS Kinetic
- Python 2.7 (required for learning code, notebooks 0 and 1)
  - TensorFlow 1.3
  - pandas
  - numpy
  - matplotlib
  - sympy
  - json
  - h5py
- Julia 0.6 (required for controls/visualization code, notebooks 2-4)
  - At the Julia prompt,  
  ```julia
    Pkg.clone("https://github.com/schmrlng/DifferentialDynamicsModels.jl")
    Pkg.clone("https://github.com/schmrlng/LinearDynamicsModels.jl")
    Pkg.update()
  ```
  - `git clone https://github.com/StanfordASL/vtd_interface.git` into the `src` directory of your ROS catkin workspace. Even if you are not using VTD to conduct human-in-the-loop trials, the `TrafficWeavingPlanner` depends on some message types defined in `vtd_interface`.
- VIRES Virtual Test Drive (VTD, required for human-in-the-loop simulation, notebook 3)
  - Contact [VIRES](https://vires.com/vtd-vires-virtual-test-drive/) for a driving simulator license.
  - The VTD project defining the highway on-ramp/off-ramp scenario used in this work may be accessed here: http://stanford.edu/~schmrlng/TrafficWeavingVTDProject.zip; we are not sure how portable these projects are so please file an issue if you are having difficulties loading up `TrafficWeavingVTDProject.vpj` contained within that zip file.
