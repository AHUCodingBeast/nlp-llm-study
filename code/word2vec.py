#导入fasttext
import fasttext
def dm_fasttext_train_save_load():
    # 使用train_unsupervised（无监督训练方法）训练词向量
    # 使用重新格式化的句子文件，每行一个句子
    mymodel = fasttext.train_unsupervised("./data/fil9_sentences.txt")

    # 保存已经训练好的词向量
    # mymodel.save_model("./data/fil9.bin")
    # 载入已经训练好的词向量
    # mymodel.load_model("./data/fil9.bin")

dm_fasttext_train_save_load()