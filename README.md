# word2vec_non-framework
外部のフレームワークを用いずに自然言語処理のディープラーニングのモデルを作成
word2vecと呼ばれるモデルの中でも複数の単語から一つの単語を推定するCBOWモデルを作成した。
また、共起行列も作成可能である

オライリー社の出版している『ゼロから作るDeepLearning② 自然言語処理編』を読み、多少の変更を加えながらフレームワークを作成している。
それが1.pyに保存されている

（https://raw.githubusercontent.com/tomsercu/lstm/master/data/）　
このURL内の各データから学習をさせている(13~23行を参照)


hidden_size = 隠れ層の数

batch_size = 一回の学習で用いるデータのサイズ

max_epoch = 何周データを学習させるか


cbow_params.pklに各単語に対するベクトルが保存される。

また学習後に損失関数のグラフが作成される。



# requirement
GPUを用いて学習させることを前提にしており、cupyx、cupyが必要である




