import MeCab

mecab = MeCab.Tagger("-Owakati")
text = "吾輩は猫である。名前はまだ無い。"
result = mecab.parse(text)
print(result)