class Lang:
    # 句子开始标志
    SOS_token = 0
    # 句子结束标志
    EOS_token = 1

    def __init__(self, name):
        """初始化语言对象
        参数:
            name: 语言名称（如'eng'表示英语，'fra'表示法语）
        """
        self.name = name
        self.word2index = {"SOS":Lang.SOS_token, "EOS":Lang.EOS_token}
        self.index2word = {Lang.SOS_token: "SOS", Lang.EOS_token: "EOS"}  # 索引到词汇的映射字典
        self.n_words = 2  # 词汇总数，初始为2（SOS和EOS已占用）

    def addWord(self, word):
        """添加词汇到映射表
        参数:
            word: 要添加的词汇
        """
        # 如果词汇不在映射表中，则添加它
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1  # 词汇计数加1

    def addSentence(self, sentence):
        """将句子拆分为词汇并添加到映射表
        参数:
            sentence: 要处理的句子
        """
        # 按空格分割句子为词汇列表
        for word in sentence.split(' '):
            self.addWord(word)
