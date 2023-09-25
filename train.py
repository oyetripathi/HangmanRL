from env import Hangman
from dqn import Agent, Trainer
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

env = Hangman(
    word_src=config["word_src"],
    num_env=config["num_env"],
    max_lives=config["max_lives"],
    win_reward=config["win_reward"],
    correct_reward=config["correct_reward"],
    repeated_guessing_penalty=config["repeated_guessing_penalty"],
    lose_reward=config["lose_reward"],
    false_reward=config["false_reward"],
    verbose=config["env_verbose"]
)

agent = Agent(
    vocab_size=config["vocab_size"],
    embed_size=config["embed_size"],
    num_heads=config["num_heads"],
    key_dim=config["key_dim"],
    maxlen=env.maxlen,
    base_model_wts=config["base_model_wts"]
)

trainer = Trainer(
    gamma=config["gamma"],
    epsilon=config["epsilon"],
    epsilon_min=config["epsilon_min"],
    epsilon_max=config["epsilon_max"],
    epsilon_random_frames=config["epsilon_random_frames"],
    epsilon_greedy_frames=config["epsilon_greedy_frames"],
    max_memory_length=config["max_memory_length"]
)

for i in range(config["train_steps"]):
    print("Train Step: ", i+1)
    trainer.train(agent, env, config["train_record_steps"], config["epochs"])
    avg_rew, win_per = trainer.eval(agent, env, config["eval_record_steps"])
    if ((i+1) % config["save_steps"]) == 0:
        print("Saving Model...")
        agent.save_target_model("target_model_wts_2x32x256.h5")
    print("Average Reward: ", avg_rew)
    print("Win Percentage: ", win_per)
    print('\n')

# agent.load_target_model("target_model_wts_2x32x256.h5")
# rew, win = trainer.eval(agent, env, config["eval_record_steps"])
# print(rew, win)