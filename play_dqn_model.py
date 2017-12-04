import gym
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input, Lambda, merge
from keras.optimizers import RMSprop
from keras.models import Model
from skimage.transform import resize
from skimage.color import rgb2gray
from keras import backend as K
from data.env import Env
from sum_tree import SumTree

EPISODES = 50000


class TestAgent:
    def __init__(self, action_size, env):
        self.env = env
        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.model = self.build_model()

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        input = Input(shape=self.state_size)
        shared = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
        shared = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(shared)
        shared = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(shared)
        flatten = Flatten()(shared)

        # network separate state value and advantages
        advantage_fc = Dense(512, activation='relu')(flatten)
        advantage = Dense(self.action_size)(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(self.action_size,))(advantage)

        value_fc = Dense(512, activation='relu')(flatten)
        value = Dense(1)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)

        # network merged and make Q Value
        q_value = merge([value, advantage], mode='sum')
        model = Model(inputs=input, outputs=q_value)
        model.summary()

        return model

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= 0.01:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def load_model(self, filename):
        self.model.load_weights(filename)


def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    env = Env()
    agent = TestAgent(action_size=6, env=env)
    agent.load_model("./save_model/supermario_per.h5")

    for e in range(EPISODES):
        done = False
        max_x = 0
        now_x = 0
        hold_frame = 0
        before_max_x = 200

        start_position = 500
        step, score = 0, 0
        observe = env.reset(start_position=start_position)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        action_count = 0
        real_action, action = 0, 0

        while not done:
            step += 1

            action = agent.get_action(history)
            if action == 0:
                real_action = 0
            elif action == 1:
                real_action = 3
            elif action == 2:
                real_action = 4
            elif action == 3:
                real_action = 7
            elif action == 4:
                real_action = 8
            else:
                real_action = 11

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, clear, max_x, timeout, now_x = \
                env.step(real_action)

            if now_x >= 8776:
                reward += 200
                done = True

            if done and now_x < 8776:
                reward = -30

            reward /= 100
            # reward = np.clip(reward, -1., 1.)
            print(now_x)
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            score += reward

            history = next_history

            if done:
                print("episode:", e, "  score:", score)

