import math, time, os, random, pygame, time
import tensorflow as tf
import keras, cv2
from FlappyQt import Pipe, Bird, Base, BG_IMG, STAT_FONT
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.applications.xception import Xception
from keras import backend as backend
from collections import deque
import numpy as np
from tqdm import tqdm
from PIL import Image
#tf.compat.v1.disable_eager_execution()

Dir = os.path.dirname(__file__)

MIN_REPLAY_MEMORY_SIZE = 500
REPLAY_MEMORY_SIZE = 2_500
MODEL_NAME = "Easy"
MINIBATCH_SIZE = 2
UPDATE_TARGET_EVERY = 5
DISCOUNT = 0.99
MIN_REWARD = -1000
#LOAD_MODEL = None
LOAD_MODEL = f"models/{MODEL_NAME}"
AVG_MODEL = False # prediction - avg of best model and current model 
RETURN_IMAGE = True

REPLAY = True # if cli or graphics
EPISODES = 40_000
SAVE_MODEL_EVERY = 200
RENDER_EVERY = 1 # if replaying how often - savescpu

epsilon = .9
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


SHOW_PREVIEW = False # drawing on the surface?
SHOW_PREVIEW_EVERY = 1
AGGREGATE_STATS_EVERY = 100
WIN_WIDTH = 500
WIN_HEIGHT = 800
pygame.init()
pygame.font.init()
pygame.display.set_caption("Flappy Qt")
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
"""

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate XGB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
"""
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0") # 
graph = tf.Graph()
#graph =  tf.compat.v1.get_default_graph()
    
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=70)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options=gpu_options), graph = graph) 
tf.compat.v1.keras.backend.set_session(sess)

if not os.path.isdir('logs'):
	os.makedirs('logs')
	print("Logs folder created------------------------------")

if not os.path.isdir('models'):
    os.makedirs('models')
    print("Model folder created---------------------------")

def distance(x1, y1, x2, y2):
	d = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
	return d

class Env:
	
	

	def __init__(self):
		if REPLAY:
			self.win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
			self.clock = pygame.time.Clock()
		self.TICK_REWARD = 1
		self.DIE_PENALTY = 500
		self.PASS_REWARD = 25
		self.BETWEEN_REWARD = 2

		self.player = Bird(230, 350)
		self.pipes = [Pipe(600)]
		self.base = Base(730)

		self.pipe_ind = 0
		self.episode = 0
		self.TheModel = False
		self.BETWEEN_THRESHOLD = 30
		self.max_score = 2

		#observation =  np.array([[(self.player.y), (abs(self.player.y - self.pipes[self.pipe_ind].height))], [(abs(self.player.y - self.pipes[self.pipe_ind].bottom)), abs(self.player.x - self.pipes[self.pipe_ind].x)]])
		#print(observation.shape, "-------------------------------------")
	def reset(self):
		self.player = Bird(230, random.randrange(175, 500))
		self.pipes = [Pipe(600)]
		self.base = Base(730)
		self.score = 0
		self.pipe_ind = 0

		self.tick_reward = 0
		self.reward = 0
		self.episode_step = 1
		self.done = False

		self.jumpTotal = 1
		self.jumpPercentage = 0

		self.boundsDeath = False

		#observation =  np.array([[(self.player.y), (abs(self.player.y - self.pipes[self.pipe_ind].height))], [(abs(self.player.y - self.pipes[self.pipe_ind].bottom))], [abs(self.player.x - self.pipes[self.pipe_ind].x)]])
		observation = self.observations() 
		#print(observation.shape, "-------------------------------------")
		return observation

	def step(self, action):
		self.reward = 0
		self.episode_step += 1
		self.player.move()
		#self.base.move()

		if REPLAY:
			for event in pygame.event.get(): # otherwise no event handling would crash on an event
				if event.type == pygame.QUIT:
					pygame.quit()
					quit()
					sys.exit()
		# self.player.action(action)
		if action == 0:
			self.player.jump()
			self.jumpTotal += 1

		#tick_reward = -abs(abs((self.pipes[self.pipe_ind].height + self.pipes[self.pipe_ind].bottom)/ 2) - self.player.y)
		self.pipe_ind = 0
		if self.pipes[self.pipe_ind].height  > self.player.y > self.pipes[self.pipe_ind].bottom + self.BETWEEN_THRESHOLD: # reward if in between gap
			self.reward += self.BETWEEN_REWARD
			
		else: 
			self.tick_reward = 0
		add_pipe = False
		rem = []

		# check pipe to input values - if passed
		
		new_observation = self.observations()

		if len(self.pipes) > 1 and self.player.x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width():
			self.pipe_ind = 1  # if passed look at the next pipe

		
		for pipe in self.pipes:  # if you hit the damn pipe reset
			if pipe.collide(self.player):
				#self.reward = -self.DIE_PENALTY
				#self.reward = -abs(self.player.y - (self.pipes[self.pipe_ind].height + self.pipes[self.pipe_ind].bottom)/2) + 20
				self.reward = -distance(self.player.x, self.player.y, 
					self.pipes[self.pipe_ind].x + self.pipes[self.pipe_ind].PIPE_TOP.get_width(),(self.pipes[self.pipe_ind].height + self.pipes[self.pipe_ind].bottom)/2)

				self.done = True
			if not pipe.passed and pipe.x < self.player.x:  # if passed pipe
				pipe.passed = True
				add_pipe = True

			if pipe.x + pipe.PIPE_TOP.get_width() < 0:  # if outside screen
				rem.append(pipe)  # say goodbye - to the trashmobile
			pipe.move()

		if add_pipe: # if you made it trough pipe you get a treat
			self.reward = self.PASS_REWARD
			self.score += 1
			self.pipes.append(Pipe(600)) # add new pipe yes

		for r in rem:
			self.pipes.remove(r) # clear recycle bin

		if self.player.y + self.player.img.get_height() >= 730 or self.player.y < 0: # if it hits the floor # no going over no
			self.boundsDeath = True
			self.reward = -self.DIE_PENALTY
			#self.reward = -distance(self.player.x, self.player.y, 
					#self.pipes[self.pipe_ind].x + self.pipes[self.pipe_ind].PIPE_TOP.get_width(),(self.pipes[self.pipe_ind].height + self.pipes[self.pipe_ind].bottom)/2)
			self.done = True
		
		
			

		self.jumpPercentage = self.jumpTotal / self.episode_step
		self.reward += self.tick_reward
		if self.score >= 30 and not self.done:
				self.done = True
				if TheModel == False:
					try:
						TheModel = True
						agent.model.save(f'models/TheModel')
					except:
						print('Saved The One Model')
				elif self.score < self.max_score:
					agent.model.save(f'models/{MODEL_NAME}')
					self.max_score = self.score
			

		return new_observation, self.reward, self.done

	def render(self, episode, render = False, epsilon = 1):
		win = self.win
		win.blit(BG_IMG, (0, 0))
		self.episode = episode
		if render:
			

			for pipe in self.pipes:
				pipe.draw(win)
			self.base.move()
			self.player.draw(win)
			self.base.draw(win)
	
		text = STAT_FONT.render("Episode: " + str(self.episode), 1, (255, 255, 255)) # white
		win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10)) # always in screen

		text = STAT_FONT.render("score: " + str(round(self.score)), 1, (255, 255, 255)) # white
		win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 50)) # always in screen

		text = STAT_FONT.render("Epsilon: " + str('%.3f' % epsilon), 1, (255, 255, 255)) # white
		win.blit(text, (10 ,10)) # always in screen

		text = STAT_FONT.render("Reward: " + str(f"{round(self.reward, 2)}"), 1, (255, 255, 255)) # white
		win.blit(text, (10 ,50)) # always in screen

		#pygame.draw.line(win, (255, 0, 0), (self.player.x, self.player.y), (self.pipes[self.pipe_ind].x + self.pipes[self.pipe_ind].PIPE_TOP.get_width(), (self.pipes[self.pipe_ind].height + self.pipes[self.pipe_ind].bottom)/2))
		# text = STAT_FONT.render("Alive: " + str(alive), 1, (255, 255, 255)) # white
		# win.blit(text, (10 ,90)) # always in screen
		pygame.display.update()

	def observations(self):
		if RETURN_IMAGE:
			rbg_image = pygame.surfarray.array3d(self.win)
			rgb_weights = [0.2989, 0.5870, 0.1140]
			grayscale_image = np.dot(rbg_image[...,:3], rgb_weights)
			return pygame.surfarray.array3d(self.win)
			#print(grayscale_image.shape)
			#return grayscale_image
		else:
			return np.array([[(self.player.y), (abs(self.player.y - self.pipes[self.pipe_ind].height))], [(abs(self.player.y - self.pipes[self.pipe_ind].bottom)), abs(self.player.x - self.pipes[self.pipe_ind].x)]], ndmin=2)

class ModifiedTensorBoard(TensorBoard):

	# Overriding init to set initial step and writer (we want one log file for all .fit() calls)
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.create_file_writer(self.log_dir)
		# self.writer = tf.summary.FileWriter(self.log_dir) # version 1 tf
		#self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)
		#self._log_write_dir = f"{self.log_dir}\\{MODEL_NAME}"
		self._log_write_dir = self.log_dir + "/"
		print(self._log_write_dir, self.log_dir)
		#print(os.path.join(self.log_dir, MODEL_NAME))
	# Overriding this method to stop creating default log writer
	def set_model(self, model):
		pass

	# Overrided, saves logs with our step number
	# (otherwise every .fit() will start writing from 0th step)
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	# Overrided
	# We train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	# Overrided, so won't close writer
	def on_train_end(self, _):
		pass

	# Custom method for saving own metrics
	# Creates writer, writes custom metrics and closes writer
	def update_stats(self, **stats):
		self._write_logs(stats, self.step)
		# self.compat.v1._write_logs(stats, self.step)

	def _write_logs(self, logs, index):
		with self.writer.as_default():
			for name, value in logs.items():
				tf.summary.scalar(name, value, step=index)
				self.step += 1
				self.writer.flush()


class DQNAgent:

	def __init__(self):

		# main model .fit every step
		self.model = self.create_model()

		# target model .predict every step
		self.target_model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		# deque is a type of list[]
		PATH = '/Models'
		# self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
		self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}")
		# self.tensorboard = ModifiedTensorBoard(MODEL_NAME, log_dir="{}logs/{}-{}".format(PATH, MODEL_NAME, int(time.time()))) # version 2.0
		self.target_update_counter = 0

	def create_model(self):
		if LOAD_MODEL is not None:
			print(f"loading {LOAD_MODEL}")
			model = load_model(LOAD_MODEL)
			print(f"Model {LOAD_MODEL} loaded succesfully")
		elif RETURN_IMAGE:
			base_model = Xception(weights=None, include_top=False, input_shape = (WIN_WIDTH, WIN_HEIGHT, 3)) # modying base inputs
			x = base_model.output
			x = GlobalAveragePooling2D()(x)

			predictions = Dense(2, activation="linear")(x) # 3 options
			model = Model(inputs= base_model.input, outputs=predictions)
			model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
			model.summary()

		else:
			model = Sequential()
			#model.add(Conv2D(256, ( 3,3 ), input_dim = 4))
			#model.add(Conv2D(256,(1, 1), input_shape = (2, 2)))
			#model.add(MaxPooling2D( 2, 2))
		
			#model.add(keras.Input(batch_shape = (2, 2) ) )
			#model.add(Conv2D(128,(1, 1)))
			model.add(Dense(64, input_shape = (2, 2)))
			#model.add(Activation('relu'))
			#model.add(Flatten())
			

			model.add(Dense(32))
			model.add(Activation('relu'))
			
			#model.add(MaxPooling2D( 2, 2))
			model.add(Dense(16))
			model.add(Activation('relu'))

			model.add(Dense(8))
			model.add(Activation('relu'))
			
			model.add(Flatten())
			model.add(Dense(2, activation='softmax'))
			#mse
			model.compile(loss = "categorical_crossentropy", optimizer= Adam(lr=0.001), metrics=['accuracy'])
			model.summary()
		return model

	def update_replay_memory(self, transition):
		self.replay_memory.append(transition)


	def get_qs(self, state):
		#return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
		#return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
		#return self.model.predict(np.array(state))[0]
		
		state = self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
		return state
		#return tf.convert_to_tensor(state, dtype=np.float32)
	def train(self, terminal_state, step):
		if len(self.replay_memory) <= MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

		current_state = np.array([transition[0] for transition in minibatch]) /255
		current_qs_list = self.model.predict(current_state) # the crazy model fitted on every step
		
		

		new_current_states = np.array([transition[3] for transition in minibatch]) /255# grabbing new current state
		future_qs_list = self.target_model.predict(new_current_states)
		
		

		X = []
		y = []

		for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			X.append(current_state)
			y.append(current_qs)

		# only Fit the model if minibatch is full
		self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
		#, callbacks=[self.tensorboard] if terminal_state else None
		# updating to determine if we want to update target model yet
		if terminal_state:
			self.target_update_counter +=1
		# update the 'STABLE' model
		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0

		replay_memory = []

agent = DQNAgent()
env = Env()

ep_rewards = [0]
ep_scores = [0]
ep_jumpPercentage = [1] # stats later for tensorlog
ep_boundsDeathCount = 0


# load best model
best_model = agent.create_model()
best_model.set_weights(agent.target_model.get_weights())
best_score = -500
if LOAD_MODEL is not None:
	print(f"loading {LOAD_MODEL}")
	best_model = load_model(f'{LOAD_MODEL}-1')
	print(f"Best Model {LOAD_MODEL} loaded succesfully")
else:
	best_model = agent.create_model()
	best_model.set_weights(agent.target_model.get_weights())


#average_reward, min_reward, max_reward =
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episode'):

	# Update tensorboard step every episode
	#agent.tensorboard.step = episode

	# Restarting episode - reset episode reward and step number
	episode_reward = 0
	step = 1

	# Reset environment and get initial state
	current_state = env.reset()

	# Reset flag and start iterating until episode ends
	done = False
	while not done:

		# This part stays mostly the same, the change is to query a model for Q values
		if np.random.random() > epsilon:
			# Get action from Q table
			if AVG_MODEL: # the best model and current model averaged out
				action = agent.get_qs(current_state)
				bestaction = best_model.predict(np.array(current_state).reshape(-1, *current_state.shape)/255)[0]
				average = (action + bestaction) / 2
				action = np.argmax(average)
			else:
				action = np.argmax(agent.get_qs(current_state))
		else:
			# Get random action
			action = random.randint(0, 1)

		new_state, reward, done = env.step(action)

		# Transform new continous state to new discrete state and count reward
		episode_reward += reward

		#if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
			#env.render(episode)

		if REPLAY:
			if not episode % SHOW_PREVIEW_EVERY:
				env.render(episode, True, epsilon)
			else:
				env.render(episode, False, epsilon)
			#env.render(episode)

		
		# Every step we update replay memory and train main network
		agent.update_replay_memory((current_state, action, reward, new_state, done))
		agent.train(done, step)

		current_state = new_state
		step += 1

	# Append episode reward to a list and log stats (every given number of episodes)
	ep_rewards.append(episode_reward)
	ep_scores.append(env.score)
	ep_jumpPercentage.append(env.jumpPercentage)
	if env.boundsDeath:
		ep_boundsDeathCount += 1
	if not episode % AGGREGATE_STATS_EVERY or episode == 1:
		average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
		min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
		max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

		average_score = sum(ep_scores[-AGGREGATE_STATS_EVERY:])/len(ep_scores[-AGGREGATE_STATS_EVERY:])
		min_score = min(ep_scores[-AGGREGATE_STATS_EVERY:])
		max_score = max(ep_scores[-AGGREGATE_STATS_EVERY:])

		average_jumpPercentage = sum(ep_jumpPercentage[-AGGREGATE_STATS_EVERY:])/len(ep_jumpPercentage[-AGGREGATE_STATS_EVERY:])
		min_jumpPercentage = min(ep_jumpPercentage[-AGGREGATE_STATS_EVERY:])
		max_jumpPercentage = max(ep_jumpPercentage[-AGGREGATE_STATS_EVERY:])

		print(f"Min reward: {min_reward}, Avg reward: {average_reward}, Max reward: {max_reward}")

		print(f"Min score: {min_score}, Avg score: {average_score}, Max score: {max_score}")

		print(f"Min jumpPercentage: {round(min_jumpPercentage, 2)}, Avg jumpPercentage: {round(average_jumpPercentage, 2)}, Max jumpPercentage: {round(max_jumpPercentage, 2)}")

		print(f"Out of bounds Death percentage: {round(ep_boundsDeathCount/ AGGREGATE_STATS_EVERY, 2)}")

		print(f"Epsilon: {round(epsilon, 4)}")
		
		if best_score < average_score:
			best_score = average_score
			print(f"Best Model so far")
			best_model.set_weights(agent.target_model.get_weights())
			best_model.save(f'models/{MODEL_NAME}-1')

		t = time.localtime()
		current_time = time.strftime("%H:%M:%S", t)
		print(f"Time : {current_time}")
		#agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

		ep_rewards = [0]
		ep_scores = [0]
		ep_jumpPercentage = [1]
		ep_boundsDeathCount = 0 
		# Save model, but only when min reward is greater or equal a set value
		if min_reward >= MIN_REWARD or episode % SAVE_MODEL_EVERY == 0:
			print(f"Saved Model {MODEL_NAME} at Episode {episode}")
			#agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
			#agent.model.save(f'{Dir}\\models\\{MODEL_NAME}')
			agent.model.save(f'models/{MODEL_NAME}')


	# Decay epsilon
	if epsilon > MIN_EPSILON:
		epsilon *= EPSILON_DECAY
		epsilon = max(MIN_EPSILON, epsilon)
		