---

pagetitle: "Deep Learningのプロジェクトから学んだこと | Chief Blog"
title: Deep Learningのプロジェクトから学んだこと
date: 2020-01-07T03:03:38-05:00
featuredImage: /img/result.png
katex: true
description: LSTMを用いた株価予測Botを作りました。今回のDeep Learningのプロジェクトから学んだことをまとめていきます。
Tags: 
- DL, Stock
Categories:
- Project
draft: false
summary: LSTMを用いた株価予測Botを作りました。今回のDeep Learningのプロジェクトから学んだことをまとめていきます。

---

## LSTMを用いた株価予測Bot

今回は、LSTMを用いた株価予測Botを作りました。Deep learningを用いたプロジェクトは初めてということもあり、Jonathan Huiさんのブログ(https://medium.com/@jonathan_hui/how-to-start-a-deep-learning-project-d9e1db90fa72)を参考にしながらプロジェクトを進めました。

## プロジェクトの目的

今回このプロジェクトを選んだのは、実際にDeep Learningプロジェクトを始めるに至って、どのプロセスが一番のボトルネックになるかを自分の目で確かめたかったからです。よくブログや掲示板などを見ていると、Feature EngineeringやData PreprocessingがMLのプロジェクトの60%以上を占めると聞きます。僕自身、ゼロから企画してプロジェクトを始めたことはなかった（既にPreprocessedされたDataを扱うことが多かった）ので、これをきっかけにDeep Learning Projectの大変さを体験してみたかったのです！

したがって、今回の目的はProductionで使えるようなAIを作る事ではありません。とはいいつつも、プロジェクトを行う以上何か役に立つものを作りたかったので、そこまで複雑そうではない株式予測Botを作ることに決めましたのです。

大まかな流れとして、以下のようにプロジェクトを進めました。実際にはこのように綺麗な感じではなく、各プロセスを行ったり来たりしています。右のカッコは各プロセスにかかったおおよその時間です。

1. Read Papers (1 Day)
2. Establish Project’s Goal（2 hours）
3. Gather Public Data（1 hour）
4. Data Preprocessing (2 Days)
5. Build Multiple Models (2 Day)
6. Debug! Debug! Debug!（2 days）

合計で、5日かけて5つのモデルを学習することができました。モデルは、複数のInput Features（Closing Price, Opening Price, High Price, Low Price, Volumeなど）を元に次の日の株価（Closing Price, Opening Price, High Price, Low Price）を予測するというものです。

Local上でゆっくりモデルを学習させる時間がなかったこともあり、適当な時間で学習をストップしてBenchmarkを比べています。

全てのモデルが均等にBenchMarkと比べられるように、Baselineはt+1のPriceがtのPriceと同じと予想した時の誤差で計算しています。つまり、Resultがプラスであるならば、単純に1日前と同じ価格で予想した時よりも高い精度であると言えます。

$$ Result = \dfrac{ Baseline - Predicted }{Baseline} $$
$$ Predicted = Predicted_{t+1} - Target_{t+1} $$
$$ Baseline = Target_{t} - Target_{t+1} $$

## 結果

次に最終的な結果です。学習したモデルは五種類あり、大きく変えたParameter/Model Designとしては次の5つがあります。

- Input Feature
- Loss Function (MSE + Directional Loss)
- Transformer vs LSTM
- Window Length（次の日(t)の株価を予測するために、どこまで遡るか(t-n)）
- Positional Embedding On/Off

データはKaggle(https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)から拾ってきており、アップルの株価のみを学習に使いました。 Input dataは全て、window length毎にMinMaxでnormalizeされており、permuteされた形でモデルにインプットされています。OptimizerはSGDを使用しています。Categorical valueは全てembeddingに変換されて、inputにconcatenateしています。"Day of Week"は曜日を指しています。このブログは、細かいテクノロジーのことを説明するブログではないので詳細は省きますが、今回のプロジェクトのソースコードは全てGithub(https://github.com/26keisuke/stock_predict)にアップロードしてあります。

```
Stock_Model_0: 

Architecture: LSTM
Input: Close, Open, High, Low
Window Length: 60
Epochs: 35
Result: .17507
```
```
Stock_Model_1: 

Architecture: LSTM
Input: Close, Open, High, Low
Window Length: 120
Epochs: 97
Result: .288717
```
```
Stock_Model_2: 

Architecture: LSTM
Input: Close, Open, High, Low, Month, Year, Day_of_Week, Hour
Window Length: 120
Epochs: 13
Result: .12283
```
```
Stock_Model_3: 

Architecture: LSTM
Input: Close, Open, High, Low, Month, Year, Day_of_Week, Hour, Positional Embedding
Window Length: 120
Epochs: 31
Result: .20049
```
```
Stock_Model_4: 

Architecture: Transformer
Input: Close, Open, High, Low, Month, Year, Day_of_Week, Hour, Positional Embedding
Window Length: 120
Epochs: 31
Result: -1.8893
```

学習時間が違うので一概には言えませんが、予想に反して一番単純なLSTM（Stock_Model_1）が一番正確だと言うことがわかりました。正直これは意外でした。株価の予想の方向（上昇か下降か）に基づいたcostや、より多くの情報をinputに加えてみたりしましたが、どれも精度歯落ちてしまいました。

このモデルがどれくらい正確かというと、下のtest dataの結果が参考になると思います。赤が本来の株価、青がLSTMが予測したt+1の株価です。

{{< figure src="/img/result.png" >}}

ざっとみると、t+1の値がtよりかけ離れていてもt-120までの値が安定しているならば、割と良い予測（t = 0~300までの部分）ができているように思います。しかしt > 300以降は、株価のスイングとともに徐々に予測が乖離していっています。もしかしすると**Appleの株価自体が比較的上昇し続けているために、このような動きをするデータが足りなかったのかもしれません。**とすると、まだまだ改善余地はあると思います。

Company/ETF embeddingを足したりtraining dataを増やしたりと、この他にも試したいことはたくさんありますが、今回はここらへんで一旦切り上げようと思います。

## プロジェクトから学んだこと

最後に、自分がこのプロジェクトから学んだことをまとめていきます。

- モデルが増えるにつれて、Data Preprocessingが非常に複雑になっていく

モデルによってはinputのサイズを変えなくてはいけないものもあるので、どのようにしてgenericなpreprocessor作っていくのかばかり考えていました。結果的に、長く複雑なコードになってしまいました。

- CPUには限界がある

待つ時間が非常にもったいない。

- Baselineを設定するのが難しい

最初は不完全なBaselineを元に複数のモデルを試していたため、どうしても優劣をつけるのが難しかったです。結果、意味なく無駄に学習させてしまったモデルが4つほどありました。自分のBaselineが数学的に正しいかは置いといて、絶対的にモデルを比較できる数字を用意することで多くの発見が生まれました。

- Hyperparameters & Model Designを複雑にしてしまう

しっかりと学習結果の意味を解釈しないまま新たにhyperparameterを追加したり、レイヤーを増やしたりしてもより複雑になるばかりか、精度も落ちることがほとんどでした。

- Clean Dataは悩みの種

データそのものがcleanじゃないと、garbage in garbage outで終わってしまいました。そのため、最初はデータの加工ばかりをしていました。プロジェクトの序盤でしっかりとcleanなdataを用意しておくことは非常に重要です！

- より早くミスに気づくこと

Tensorboardなどを使ってgradientやweightを可視化することによって、より早くミス（nanとかgradient vanishingとか）に気づく必要性に気付きました。

これらを一言でまとめるならば、**"Deep learningのDebugginは非常に難しい"**ということです。たった5日間の趣味程度のプロジェクトでしたが、非常に学ぶことは多かったです。このような解決すべき"自分のニーズ"を見つけた今、今後はDeep learning周りのtoolを一通りみていこうと思います。