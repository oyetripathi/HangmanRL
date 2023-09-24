import numpy as np

class Hangman(object) :
    def __init__(self , 
                 word_src, 
                 num_env = 2,
                 max_lives = 5 , 
                 win_reward = 100,
                 correct_reward = 20,
                 repeated_guessing_penalty = -1000,
                 lose_reward = -100, 
                 false_reward = -50,
                 verbose = False) :
        if type(word_src) == list :
            self.words = word_src
        else :
            with open(word_src, 'r') as f :
                self.words = f.read().splitlines()
        self.maxlen = max([len(word) for word in self.words])
        self.max_lives = max_lives
        self.num_env = num_env
        self.win_reward = win_reward
        self.correct_reward = correct_reward
        self.lose_reward = lose_reward
        self.false_reward = false_reward
        self.verbose = verbose
        self.repeated_guessing_penalty = repeated_guessing_penalty
        self.num_actions = 26
        
    def set_guess_words(self) :
        self.guess_word = np.random.choice(self.words, size=self.num_env)
        self.correct = np.zeros(shape=(self.num_env, 26))
        for idx, word in enumerate(self.guess_word):
            for char in word:
                self.correct[idx, ord(char)-ord('a')] = 1
    
    def build_gameboard(self):
        guessing_board = np.full((self.num_env, self.maxlen), '.')
        for idx, word in enumerate(self.guess_word):
            guessing_board[idx, :len(word)] = '_'
        return guessing_board

        
    def reset(self) :
        self.curr_live = np.array([self.max_lives]*self.num_env)
        self.set_guess_words()
        self.guessing_board = self.build_gameboard()
        self.correct_guess = np.zeros(self.num_env)
        self.guessed = np.zeros(shape=(self.num_env, 26))
        self.lives = np.full(shape=self.num_env, fill_value=self.max_lives)
        self.winners = np.full(self.num_env, False)
        self.losers = np.full(self.num_env, False) 
        self.done = np.full(self.num_env, False)
        if self.verbose :
            print('Game Starting')
            print('Current live :', self.curr_live)
        return self.show_gameboard(), self.guessed
        

    def show_gameboard(self) :
        board = [''.join(list(word)).replace('.','') for word in self.guessing_board]
        if self.verbose:
            print(board)
            print()
        return board
    
    def one_hot_action(self, action):
        b = np.zeros((len(action), 26))
        for idx, char in enumerate(action):
            b[idx, ord(char)-ord('a')] = 1
        return b
    
    def step(self, action):
        if any([not char.isalpha() for char in action]):
            raise TypeError("Only Alphabets are allowed")

        action = [char.lower() for char in action]
        for i in range(self.num_env):
            if self.done[i]:
                continue
            for j in range(len(self.guess_word[i])):
                if self.guess_word[i][j]==action[i]:
                    self.guessing_board[i][j] = action[i] 
        
        action = self.one_hot_action(action)
        reward = np.zeros(shape=self.num_env) 

        illegal_moves = np.any(np.logical_and(action, self.guessed), axis=-1)
        correct_moves = np.any(np.logical_and(self.correct, np.logical_and(action, np.logical_not(self.guessed))), axis=-1)
        incorrect_moves = np.any(np.logical_and(np.logical_not(self.correct), np.logical_and(action, np.logical_not(self.guessed))), axis=-1)
        self.guessed = np.logical_or(self.guessed, action)

        reward[illegal_moves] = self.repeated_guessing_penalty
        reward[correct_moves] = self.correct_reward
        reward[incorrect_moves] = self.false_reward
        reward[self.done] = 0

        winners = np.logical_not(np.any(np.logical_xor(self.correct, np.logical_and(self.guessed, self.correct)), axis=-1))
        reward[np.logical_xor(self.winners, winners)] += self.win_reward
        self.winners = np.logical_or(self.winners, winners)

        losers = np.full(shape=self.losers.shape, fill_value=False)
        self.lives[np.logical_and(incorrect_moves, np.logical_not(self.done))] -= 1
        losers[np.where(self.lives==0)] = True #losers
        reward[np.logical_xor(losers, self.losers)] += self.lose_reward
        self.losers = np.logical_or(self.losers, losers)

        self.done = np.logical_or(self.done, self.winners)
        self.done = np.logical_or(self.done, self.losers)

        return (self.show_gameboard(), self.guessed), reward, self.done, None


        


# word_src = "words_250000_train.txt"
# max_lives = 2
# num_env = 4
# env = Hangman(word_src, num_env=num_env, max_lives=max_lives, verbose = True)

# env.reset()

# done = env.done
# print(env.done)
# while np.any(np.logical_not(env.done)) :
#     action = []
#     for i in range(num_env):
#         action.append(input('Guessing letter : '))
#     _, reward , _, _ = env.step(action)
#     print(reward)
#     print(env.done)

