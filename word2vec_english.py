import math
import random

# 1. データの準備
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog sleeps all day",
    "the quick brown fox is cunning and swift",
]


# 2. 前処理
def preprocess(corpus):
    words = []
    for sentence in corpus:
        words.extend(sentence.lower().split())
    return words


words = preprocess(corpus)
vocab = list(set(words))
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}


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

        print(f"Epoch {epoch + 1}, Loss: {total_loss}")


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
word1, word2 = "fox", "dog"
similarity = cosine_similarity(get_vector(word1), get_vector(word2))
print(f"Similarity between '{word1}' and '{word2}': {similarity}")

# 全単語のベクトルを表示
for word in vocab:
    print(f"{word}: {get_vector(word)}")
