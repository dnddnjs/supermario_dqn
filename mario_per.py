from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Input, Lambda, merge
from keras.optimizers import RMSprop
from keras.models import Model
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
from data.env import Env
from sum_tree import SumTree
import random

EPISODES = 50000


# 브레이크아웃에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size, env):
        self.env = env
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 400000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = Memory(400000)
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/supermario_her', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        prediction = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

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

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, done):
        # TD-error 를 구해서 같이 메모리에 저장
        target = self.model.predict([history])
        old_val = target[0][action]
        target_val = self.target_model.predict([next_history])

        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * (
                np.amax(target_val[0]))
        error = abs(old_val - target[0][action])

        self.memory.add(error, (history, action, reward, next_history, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = self.memory.sample(self.batch_size)

        errors = np.zeros(self.batch_size)
        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][1][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][1][3] / 255.)
            action.append(mini_batch[i][1][1])
            reward.append(mini_batch[i][1][2])
            dead.append(mini_batch[i][1][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            old_val = target[i][action[i]]
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]
            errors[i] = abs(old_val - target[i])

        # TD-error로 priority 업데이트
        for i in range(self.batch_size):
            idx = mini_batch[i][0]
            self.memory.update(idx, errors[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = Env()
    agent = DQNAgent(action_size=6, env=env)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        max_x = 0
        now_x = 0
        hold_frame = 0
        before_max_x = 200

        start_position = 500
        step, score, start_life = 0, 0, 5
        observe = env.reset(start_position=start_position)

        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        action_count = 0
        real_action, action = 0, 0

        while not done:
            global_step += 1
            step += 1

            # 0: stop, 3: left, 4: left jump, 7:right, 8:right jump 11: jump
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
            if clear:
                reward += 30
                done = True

            if done and not clear:
                reward = -30

            reward /= 30
            reward = np.clip(reward, -1., 1.)

            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward
            history = next_history

            if now_x <= before_max_x:
                hold_frame += 1
                if hold_frame > 2000:
                    break
            else:
                hold_frame = 0
                before_max_x = max_x

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/supermario_her.h5")
