# -*- coding: utf-8 -*-
import math
import random

import re
import MeCab

# 1. データの準備
corpus = [
    "猫は魚が好きです",
    "犬は散歩が好きです",
    "私は犬と猫が好きです",
    "彼女は魚を料理するのが上手です",
]


# 2. 前処理
# def tokenize_japanese(text):
# 簡易的な日本語分かち書き
# tokens = re.findall(
#     r"[一-龥々ぁ-んァ-ヶー]+|[ａ-ｚＡ-Ｚ０-９]+|[、。！？]|\w+", text
# )
# print(f"Tokenized: {tokens}")
# return tokens


def tokenize_japanese(text):
    mecab = MeCab.Tagger("-Owakati")
    print(mecab)
    return mecab.parse(text).strip().split()


def preprocess(corpus):
    words = []
    for sentence in corpus:
        words.extend(tokenize_japanese(sentence))
    print(words)
    return words


words = preprocess(corpus)
vocab = list(set(words))
# print(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}
# print(word_to_index)
index_to_word = {i: word for i, word in enumerate(vocab)}
# print(index_to_word)


# 3. モデルの定義
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W1 = [
            [random.uniform(-1, 1) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.W2 = [
            [random.uniform(-1, 1) for _ in range(vocab_size)]
            for _ in range(embedding_dim)
        ]

    def forward(self, input_word):
        hidden = self.W1[input_word]
        output = [
            sum(hidden[i] * self.W2[i][j] for i in range(self.embedding_dim))
            for j in range(self.vocab_size)
        ]
        return hidden, output

    def backward(self, hidden, output, target, learning_rate):
        EI = [0] * self.vocab_size
        EI[target] = 1

        # 出力層の誤差
        EO = [output[i] - EI[i] for i in range(self.vocab_size)]

        # 隠れ層の誤差
        EH = [
            sum(EO[j] * self.W2[i][j] for j in range(self.vocab_size))
            for i in range(self.embedding_dim)
        ]

        # 重みの更新
        for i in range(self.embedding_dim):
            for j in range(self.vocab_size):
                self.W2[i][j] -= learning_rate * EO[j] * hidden[i]

        return EH

    def update_input_weights(self, input_word, EH, learning_rate):
        for i in range(self.embedding_dim):
            self.W1[input_word][i] -= learning_rate * EH[i]


# 4. 学習
def train_skip_gram(model, words, window_size, learning_rate, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for i, word in enumerate(words):
            input_word = word_to_index[word]
            for j in range(
                max(0, i - window_size), min(len(words), i + window_size + 1)
            ):
                if i != j:
                    target_word = word_to_index[words[j]]
                    hidden, output = model.forward(input_word)

                    # ソフトマックス関数
                    exp_output = [math.exp(o) for o in output]
                    softmax_output = [e / sum(exp_output) for e in exp_output]

                    # 損失の計算 (クロスエントロピー)
                    loss = -math.log(softmax_output[target_word] + 1e-10)
                    total_loss += loss

                    EH = model.backward(
                        hidden, softmax_output, target_word, learning_rate
                    )
                    model.update_input_weights(input_word, EH, learning_rate)

        # print(f"エポック {epoch + 1}, 損失: {total_loss}")


# 5. モデルの初期化と学習
vocab_size = len(vocab)
embedding_dim = 10
window_size = 2
learning_rate = 0.01
num_epochs = 1000

model = SkipGramModel(vocab_size, embedding_dim)
train_skip_gram(model, words, window_size, learning_rate, num_epochs)


# 6. 結果の確認
def get_vector(word):
    return model.W1[word_to_index[word]]


def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (magnitude1 * magnitude2)


# 単語ベクトルの類似度を計算
word1, word2 = "犬", "猫"
similarity = cosine_similarity(get_vector(word1), get_vector(word2))
print(f"'{word1}'と'{word2}'の類似度: {similarity}")

# 全単語のベクトルを表示
for word in vocab:
    print(f"{word}: {get_vector(word)}")
