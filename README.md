# Kane and Abel: A.I's that play games.

This README describes running the project in a Unix based environment.

1. clone this project from Github by doing `git clone https://github.com/saintograph/cm3070_final_project`
2. install Python and PyTorch. The version of Python used in the development of this project is Python `3.8.10`. Instructions for installing PyTorch can be found here: https://pytorch.org/get-started/locally/
3. install all packages with `python -m pip install -r requirements.txt`. This installs all of the packages required to run this project.

### Running demos

This section describes how to run the demos with saved weights.

1. download the saved weights from https://www.dropbox.com/sh/l1lmlyxjuxb4glh/AAAvHKuZhmSj2NJfeXaEW37Da?dl=0 and copy the files to the 'weights' directory. Do not _unzip_ the files.
2. run the PPO agent demonstration with `python demo_ppo.py`.
3. run the A2C agent demonstration with `python demo_a2c.py`.
4. run the FSM agent with `python fsm.py`.

### Running Optuna trials

Optuna trials were used to find optimal hyperparameter settings within a search space. To view results, a tool such as https://sqlitebrowser.org/ could be used to browse the hyperparameter values stored in `trials_{ALGORITHM}.db` after a successful trial. There are two existing `.db` files submitted, `trials_A2C.db` and `trials_PPO.db`.

1. run trials with `python trials.py PPO 10000`. The preceeding command runs 100 trials for the PPO algorithm for 10000 time-steps each trial. Note the upper-case characters when specifying an algorithm.
2. for A2C, run `python trials.py A2C 10000`.
3. a SQLite database is created in the root folder when a trial is initiated.
4. the search space for PPO and A2C can be defined in the `params.py` file.
5. at the end of 100 trials, the trial which produced the best reward values will be displayed on the command line and accessible by browsing the `trial_values` table in the generated `.db` file.


### Training Agents

After acquiring optimal hyperparameter settings, it's time to train the agents! Each model is trained 500,000 times at 5 difficulty levels (i.e. total = 5 * 500000).

1. to train the PPO model, use `python main_ppo.py`
2. to train the A2C model, use `python main_a2c.py`
3. logs are saved to the `logs` folder and training weights to the `train` folder.

### Evaluation Kane and Abel

Kane and Abel are evaluated with simple measurements - the agent which performs best over 100 episodes is the winner!


1. run `python evaluate.py`. The script loads weights from the training phase and generates a CSV file.
2. to generate the P-value and mean for the training phase, run `python stats.py` to generate results for both PPO and A2C.

### Notes

The script for training YOLOv5 was not included with the project, but some sample training and validation data can be found in the `sample_yolo_training_images` folder. All of the training data was annotated with [LabelImg](https://github.com/heartexlabs/labelImg).

To train the a YOLOv5 object detection model, instructions can be found here: https://github.com/ultralytics/yolov5

1. run `git clone https://github.com/ultralytics/yolov5` 
2. copy the `doom.yaml` file from `sample_yolo_training_images` to the root folder of the cloned repo
3. create a `demon` folder, and copy the contents of the `train` and `val` folder to it.
4. run `python train.py --batch 64 --epochs 100 --data doom.yaml --weights yolov5s.pt` to generate weights

The Path Planning implementation of A* search was reused from https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py with small modifications.

To run the algorithm, uncomment either line 237 or 238 in `PathPlanning/a_star.py`.

Simply run the script with 
1. `cd PathPlanning`
2. `python a_star.py`.