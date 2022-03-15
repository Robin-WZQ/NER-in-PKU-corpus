'''
@ file function: 词嵌入向量
@ author: 王中琦
@ date: 2022/3/15
''' 

from gensim.models.word2vec import Word2Vec
from numpy import array


def seg_word(all) -> list:
    '''
    去除停用词
    '''
    stopwords = set()
    with open('stopwords.txt', 'r', encoding='utf-8') as fr:
        for i in fr:
            stopwords.add(i.strip())
    return list(filter(lambda x: x not in stopwords, all))


def count_feature(file_name) -> array:
    '''
    文件主函数，计算文本词语特征
    '''
    
    all_sentence = []

    with open(file_name, 'r', encoding='utf-8') as inp:
        for line in inp.readlines():  # 读入每行文字
            all_words = []
            line = line.rstrip("\n")
            words = line.split(" ")
            words.pop()

            for word in words:
                all_words.append(word.split("/")[0])
            all_words = seg_word(all_words)
            all_sentence.append(all_words)

    
    print("读入数据完毕")

    return all_sentence


def model_train(token_list):
    num_features = 500
    min_word_count = 1 
    num_workers = 1
    window_size = 3
    subsampling = 1e-3

    model = Word2Vec(
        token_list,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=window_size,
        sample=subsampling,
        epochs=100,
        sg=1
    )

    model.init_sims(replace=True)
    model_name = "my_word2vec_skip"
    model.save(model_name)

    return True


def main():
    file_name = file_name = "./data/train/name_train.txt"
    token_list = count_feature(file_name)

    if(model_train(token_list)):
        print("训练完成")

    model = Word2Vec.load("my_word2vec_skip")

    for e in model.wv.most_similar(positive=['泽民'], topn=10):
        print(e[0], e[1])


if __name__ == "__main__":
    main()
