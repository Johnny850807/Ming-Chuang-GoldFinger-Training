import jieba
import numpy as np
import collections
import operator
from keras.models import Sequential
from keras.layers import Dense

puncts = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

jieba.set_dictionary('dict.txt.big')


def num_to_onehot(num):
    y = [0] * MingChuanGoldMiner.OUTPUT_SIZE
    y[num] = 1
    return y


class MingChuanGoldMiner:
    CATEGORIES = ['好過', '不好過', '收穫多', '提供資訊', '麻煩', '提問']
    WORDS_FILE_PATH = 'words_output.txt'
    OUTPUT_SIZE = len(CATEGORIES)

    def __init__(self, word_index_dict=None, output_words=False):
        self.words_count_dict = collections.defaultdict(int)
        self.output_words = output_words
        self.word_index_dict = word_index_dict or dict()

        self.trains = None
        self.labels = None
        self.train_data = None
        self.train_labels = None

        self.test_data = None
        self.test_labels = None

    def read_data_labels_from_file(self, file_path):
        with open(file_path, 'r+', encoding='utf-8') as f:
            data = f.readlines()
            trains = []
            labels = []
            for comment in data:
                segs = comment.strip().split(' ')  # ... sentence segments ... label
                category_num = int(segs[-1])
                if category_num != 6:
                    labels.append(num_to_onehot(category_num))
                    sentence = ''.join(comment.strip().split(' ')[-2:-1])
                    trains.append(self.cut_sentence_to_words_and_feed_the_words_dict(sentence))

            self.trains = np.array(trains)
            self.labels = np.array(labels)

    def setup_word_index_dict_if_not_exists(self):
        if not self.word_index_dict:
            # sort all word by its frequency
            sorted_words = [(k, v) for k, v in (sorted(self.words_count_dict.items(), key=operator.itemgetter(1)))
                            if k not in puncts]
            all_words = [word for word, count in sorted_words]
            if self.output_words:
                with open(MingChuanGoldMiner.WORDS_FILE_PATH, 'w+', encoding='utf-8') as f:
                    f.writelines(word + '\n' for word in all_words)

            for i in range(len(all_words)):
                self.word_index_dict[all_words[i]] = i

    def separate_training_data_test_data(self):
        count = self.trains.shape[0]
        self.train_data = np.array(
            [self.normalize_words_to_word_count_list(words) for words in self.trains[0: int(count * 0.8)]])
        self.train_labels = self.labels[0: int(count * 0.8)]
        self.test_data = np.array(
            [self.normalize_words_to_word_count_list(words) for words in self.trains[int(count * 0.8):]])
        self.test_labels = self.labels[int(count * 0.8):]

        assert self.train_data.shape[1] == self.test_data.shape[1] and self.train_labels.shape[1] == \
                                                                       self.test_labels.shape[1]

    def create_NN_model(self, optimizer='rmsprop', binary=False, epochs=25, batch_size=20):
        loss = 'binary_crossentropy' if binary else 'categorical_crossentropy'
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=len(self.word_index_dict)))
        model.add(Dense(MingChuanGoldMiner.OUTPUT_SIZE, activation='softmax'))
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=['accuracy'])
        model.fit(self.train_data, self.train_labels,
                  epochs=epochs,
                  batch_size=batch_size)
        return model

    def data_preprocessing(self, file_path):
        self.read_data_labels_from_file(file_path)
        self.setup_word_index_dict_if_not_exists()
        self.separate_training_data_test_data()
        return self.train_data, self.train_labels, self.test_data, self.test_labels

    def start_training(self):
        best_score = -1
        best_model = None
        for i in range(100):
            print("Round: " + str(i))
            model = self.create_NN_model()
            score = model.evaluate(self.test_data, self.test_labels, batch_size=20)
            print("Score: " + str(score))
            if score[1] > best_score:
                best_model = model
                best_score = score[1]

        print("Best score: " + str(best_score))
        return best_model

    def torture_ai(self, model):
        while True:
            sentence = input("Input a sentence (e for exit): ")
            if 'e' in sentence.strip().lower():
                break
            words_input = np.array([self.sentence_to_normalized_words_count_lists(sentence)])
            y = model.predict(words_input) - 0.3
            print(str(y))
            if np.all(y <= 0):
                print("不確定...")
            else:
                print("Output: " + MingChuanGoldMiner.CATEGORIES[np.argmax(y)])

    def sentence_to_normalized_words_count_lists(self, sentence):
        ls_words = [word for word in jieba.cut(sentence, cut_all=True)]
        return self.normalize_words_to_word_count_list(ls_words)

    def cut_sentence_to_words_and_feed_the_words_dict(self, sentence):
        words = jieba.cut(sentence, cut_all=True)
        print("Comment: " + sentence + ":    ==>   ", end="")
        ls_words = []
        for word in words:
            self.words_count_dict[word] += 1
            ls_words.append(word)
            print(word + " ", end="")
        print()
        return ls_words

    def normalize_words_to_word_count_list(self, words):
        counts = np.zeros(len(self.word_index_dict))
        for word in words:
            if word in self.word_index_dict:
                index = self.word_index_dict[word]
                counts[index] += 1
        return counts


# setup own word set
word_index_dict = {}
with open(MingChuanGoldMiner.WORDS_FILE_PATH, 'r', encoding='utf-8') as f:
    words = [word.strip() for word in f.readlines() if word[0] != '#']
    for i in range(len(words)):
        word_index_dict[words[i]] = i

miner = MingChuanGoldMiner(word_index_dict=word_index_dict)
miner.data_preprocessing('labeled_comments.txt')
model = miner.start_training()
miner.torture_ai(model)
