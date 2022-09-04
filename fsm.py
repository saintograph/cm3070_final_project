from argparse import ArgumentParser
import os
from time import sleep
import vizdoom as vzd
import torch
import torchvision
import cv2
import numpy as np
from fsm_func import DoomStateMachine
from utilities import calc_area

DEFAULT_CONFIG = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
frames_per_action = 1

def fsm():
    parser = ArgumentParser()
    parser.add_argument(dest="config", default=DEFAULT_CONFIG, nargs="?")
    args = parser.parse_args()
    game = vzd.DoomGame()

    game.load_config(args.config)

    # Automap
    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)

    game.set_screen_format(vzd.ScreenFormat.BGR24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(True)

    game.set_window_visible(True)
    game.set_mode(vzd.Mode.PLAYER)
    
    game.add_available_game_variable(vzd.GameVariable.CAMERA_POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.CAMERA_POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.CAMERA_ANGLE)
    game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    game.add_available_button(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA)
    game.add_available_button(vzd.Button.MOVE_LEFT_RIGHT_DELTA)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA)

    game.add_game_args("+sv_cheats 1")
    game.add_game_args("+freelook 1")

    game.init()

    episodes = 10
    sleep_time = 0.028
    # load object detection model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = './yolov5_weights/best.pt')
    total_reward = 0
   
    for i in range(episodes):
        game.new_episode()
        # game.send_game_command("iddqd") # cheat code
        current_episode = i
        while not game.is_episode_finished():
            state = game.get_state()

            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()

            # Get screen buffer from game engine
            screen = state.screen_buffer

            # Object detection on screen buffer with trained YOLOv5 model
            results = model(screen)
            all_demons = results.xyxy[0].tolist()

            if game.get_state() == None:
                pass
            else:
                # Instantiate agent
                agent = DoomStateMachine(game) 

                if len(all_demons) > 0:
                    for i in range(len(all_demons)):
                        demon = all_demons[i]
                        main_target_total_area = calc_area(all_demons[-1][0], all_demons[-1][1], all_demons[-1][2], all_demons[-1][3])

                        # Draw bounding box around all detections with OpenCV
                        if len(demon) > 0:
                            if demon[4] > 0.40:
                                xmin = round(demon[0])
                                xmax = round(demon[2])
                                ymin = round(demon[1])
                                ymax = round(demon[3])

                                cv2.rectangle(screen, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    
                    """
                    False positives still occur, the check below verifies:
                    - object detection model's confidence is above 40%
                    - area of detection is above a certain size (i.e. enemy NPC will have total areas higher than 500)
                    """
                    if (len(all_demons) > 0) and all_demons[-1][4] > 0.40 and main_target_total_area > 500:
                            demon = all_demons[-1]

                            xmin = round(demon[0])
                            xmax = round(demon[2])
                            ymin = round(demon[1])
                            ymax = round(demon[3])

                            width = xmax - xmin
                            height = ymax - ymin

                            # turn to face enemy
                            if xmin + (width / 2) < 310:
                                    agent.move('turn_left')
                            elif xmin + (width / 2) > 330:
                                    agent.move('turn_right')
                            else:
                                agent.stop()
                                agent.shoot()

                else:
                    # No enemies, reorientate to center of corridor
                    agent.reorientate()
            
                # show Kane's screen buffer
                cv2.imshow('ViZDoom Screen Buffer', screen)
                cv2.waitKey(int(sleep_time * 1000))
        
        total_reward = game.get_total_reward() * 0.001
    
    cv2.destroyAllWindows()
    return total_reward

if __name__ == "__main__":
    fsm()