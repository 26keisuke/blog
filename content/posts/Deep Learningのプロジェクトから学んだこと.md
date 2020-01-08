---

pagetitle: "Deep Learningのプロジェクトから学んだこと | Chief Blog"
title: Deep Learningのプロジェクトから学んだこと
date: 2020-01-07T03:03:38-05:00
featuredImage: /img/result.png
katex: true
description: LSTMを用いた株価予測Botを作りました。今回のDeep Learningのプロジェクトから学んだことをまとめていきます。
Tags: 
- DL, Stock Market, Prediction
Categories:
- Project
draft: false
summary: LSTMを用いた株価予測Botを作りました。今回のDeep Learningのプロジェクトから学んだことをまとめていきます。

---

## LSTMを用いた株価予測Bot

今回は、LSTMを用いた株価予測Botを作りました。Deep learningを用いたプロジェクトは初めてということもあり、Jonathan Huiさんのブログ(https://medium.com/@jonathan_hui/how-to-start-a-deep-learning-project-d9e1db90fa72)を参考にしながらプロジェクトを進めました。

## プロジェクトの目的

今回このプロジェクトを選んだのは、Deep learningプロジェクトを進めるに至ってどの段階が一番のボトルネックになるかを自分の目で確かめたかったからです。よくブログや掲示板などを見ていると、feature engineeringやdata preprocessingなどがMLのプロジェクトの60%以上を占めると聞きます。しかし、既に用意されたデータを用いたtutorialなどではなかなかこの大変さを実感することはできませんでした。従って、ゼロからプロジェクトを企画することでdeep learning projectがどれほど難しいのか、そしてどのような所に改善の余地があるのかを見つけようと思い、始めることにしました。つまり、今回の目的はproductionで使えるようなAIを作る事ではありません。それよりも、deep learning projectを進めるに至って行き当たった障壁についてまとめていきたいと思います。

大規模なプロジェクトは避けたかったので、そこまで複雑でなさそうな株式予測Botを作ることに決めましたのです。

大まかな流れとして、以下のようにプロジェクトを進めました。実際にはこのような綺麗な流れではなく、各プロセスを行ったり来たりしています。右のカッコは各プロセスにかかったおおよその時間を示しています。

1. Read Papers (1 Day)
2. Establish Project’s Goal（2 hours）
3. Gather Public Data（1 hour）
4. Data Preprocessing (2 Days)
5. Build Multiple Models (2 Day)
6. Debug! Debug! Debug!（2 days）

合計で、5日かけて5つのモデルを学習することができました。モデルは、複数のInput Features（Closing Price, Opening Price, High Price, Low Price, Volumeなど）を元に、次の日の株価（Closing Price, Opening Price, High Price, Low Price）を予測するというものです。

次に、各プロセスで僕が感じたことを共有していきます。

## 1. Read Papers

まず、DL x Stock Predictionの分野で数十の論文を読みました。その中でも僕が面白いと思ったカテゴリーは以下の3つです。一つわかりにくかったのが、多くの論文がモデルの精度をMSEを用いて比較していため、実際に株価予測として使えるレベルなのかがよくわかりませんでした。

- Natural Language Processing
ReutersやFinancial Timesの記事を参考にするものと、StockTwitsやHotCopperなどの掲示板をリアルタイムでスクレイピングするものの二種類に分けられます。基本的なCNNやFNNを一から学習させたものがほとんどで、Bertなどの既に学習されたモデルを使ったものは1つだけでした。僕自身、Bertと使ってどれくらいの精度が出るか興味があったのですが、あまり驚くほどの結果はまだ出ていないようです。

特に何回か引用されていたのが、この論文(https://www.ijcai.org/Proceedings/15/Papers/329.pdf)でした。簡単に要約すると、各記事を(actor, action, object)という形の特殊なrepresentationに変形した上でevent representation(document embeddingのようなもの)をCNNで計算し、最後にタイムステップ分の記事を用いて再度CNNで結果を予測するというものです。Document embeddingを使うということで、少しextractive summarizationに似ていました。

- CNN with Gramian Angular Field
二つ目がGramian Angular Field(GAF)を使って、time seriesを二次元のimageに移し替えるというものです。簡単にGAFの流れを説明すると、

1. MinMaxでスケーリングする
2. タイムステップiと値xを元に、polar coordinateに移し替える
$$ \phi_{i} = \arccos{x} $$  
$$ r_{i} = radius(i/N) $$
3. 最後にinner productの代わりに 
$$ \cos(\phi_{x} + \phi_{y}) $$ 
を用いてGramian Matrixを計算する。ここでinner productを使わない理由は、2つの情報量(xとy)が1つに減ってしまうからだそうです。

ここに(https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3)綺麗なアニメーションがあるので、わかりやすいと思います。

ちなみに、imageに移し替えた後でも時間の関係性はしっかりと保持しています。画像でいうならば、左上から右下にかけて扇状に縮小していく感じです。

- RNN

最後が、普通のLSTMです。これは説明が要らないと思います。ただ古い論文が多かったので、layer normalizationなどの比較的新しい（とは言っても3,4年前くらいですが）テクニックは使われていません。


## 2. Establish Project’s Goal

論文を読む前にゴールを決めようかとも思いましたが、最近のトレンドを把握できない限り具体的な方向性がわからないため、論文を一通り読んだ後に決めました。ゴールは最初に語ったように、"deep learning projectを進めるに至って行き当たる障壁"を見つけるということです。

プロジェクトを終えて思ったのは、ゴールをただ決めるだけではなく、今回のプロジェクトで何をして何をしないかというのをイメージしておくことは必要だということです。例えば今回のプロジェクトの場合、「より多くの時間を削ってもっと精度の高いモデルを探す」という点で少し悩みました。というのも、いろんなモデルを試していくと、ある時点からどうしても反復的な作業ばかりになってしまいます。時間と費用を照らし合わせ、どこで切り上げるかをしっかり把握しておくことは重要です。

## 3. Gather Public Data

今回のプロジェクトに使ったデータはKaggle(https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)から拾ってきており、アップルの株価のみを学習に使いました。 株価のデータはネットにたくさんあったため、比較的すぐ見つけることができました。

プロジェクトを進めている時に思ったのが、Public dataの中には信頼性の低いデータもたくさんあるということです。そのため、事前にstatisticsをとって、nanの数や数字の分布などを最初に確認しといた方がいいと思います。例えば今回使ったデータセットの場合、closing priceが前日よりも「高かったデータ」「低かったデータ」「変わらなかったデータ」の数はそれぞれ20757、20551、1837でした。変化の幅の平均もそれぞれ0.0631と0.0632と、ほとんど同じです。また、find_outlierというオリジナルの関数を作って、あらかじめ株価が異常に上昇/下降しているデータを把握できるようにしました。

## 4. Data Preprocessing

今回のプロジェクトではここに一番の時間を費やしました。

まず、モデルが増えるにつれてdata preprocessingが非常に複雑になっていきます。というのも、モデルによってはinputのrepresentationを大きく変えなくてはいけないものがあります。しかし、将来作るモデルのinputデータについて深く考えていなかったので、どうしても余分な時間を使ってしまいました。結果的に長く複雑なコードになってしまいました。

もう一つが、Dirty Dataは悩みの種だということです。データそのものがcleanじゃないと"garbage in garbage out"、つまり、モデル自体も不完全で終わってしまいます。僕の場合は、40000以上のデータの内の数個の値がnanだったせいで、モデルのweightsも全てnanになってしまいました。プロジェクトの序盤でしっかりとcleanなdataを用意しておくことは非常に重要です！Tensorboardなどのツールを使ってgradientやweightを可視化するべきだったなと後悔しています。

## 5. Build Multiple Models

まず、必要なのが適切なbaselineの設定です。僕の場合、最初に不完全なbaselineを設定していたため、どうしてもモデルの優劣をつけるのが難しかったです。結果的に、無駄に学習させてしまったモデルが4つほどありました。絶対的にモデルを比較できる数字を用意することで、多くの発見が生まれました。

全てのモデルが均等にbaselineと比べられるように、baselineはt+1のpriceがtのpriceと同じと予想した時の誤差で計算しています。つまり、Resultがプラスであるならば、単純に1日前と同じ価格で予想した時よりも高い精度であると言えます。

$$ Result = \dfrac{ Baseline - Predicted }{Baseline} $$
$$ Predicted = Predicted_{t+1} - Target_{t+1} $$
$$ Baseline = Target_{t} - Target_{t+1} $$

もうひとつ行き当たった壁が、しっかりと学習結果の意味を解釈しないまま新たにhyperparameterを追加したり、レイヤーを増やしたりしてしまったことです。これによってモデルがより複雑になるだけでなく、精度も落ちることがほとんどでした。

## 結果

次に最終的な結果です。学習したモデルは五種類あり、大きく変えたParameter/Model Designとしては次の5つがあります。Local上でゆっくりモデルを学習させる時間がなかったこともあって、適当な時間で学習をストップしてモデルの精度を比べています。変えた要素は以下の通りです。

- Input Feature
- Loss Function (MSE + Directional Loss)
- Transformer vs LSTM
- Window Length（次の日(t)の株価を予測するために、どこまで遡るか(t-n)）
- Positional Embedding On/Off

Input dataは全て、window length毎（単位は分）にMinMaxでnormalizeされており、permuteされた形でモデルにインプットされています。OptimizerはSGDを使用しています。Categorical valueは全てembeddingに変換されて、inputにconcatenateしています。"Day of Week"は曜日を指しています。その他の詳細は省きますが、今回のプロジェクトのソースコードは全てGithub(https://github.com/26keisuke/stock_predict)にアップロードしてあります！

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

学習時間が違うので一概には言えませんが、予想に反して一番単純なLSTM（Stock_Model_1）が一番正確だと言うことがわかりました。株価の予想の方向（上昇か下降か）に基づいたcostや、より多くの情報（time, position）をinputに加えてみたりしましたが、どれも精度が落ちてしまっています。これらの理由として考えられるのは、「そもそもデータの数が少なすぎるせいで、MonthやHourなどはただ単に相関がないnoiseになってしまっている」という仮説です。もしLSTMのどのlayerがactivateされているのか可視化できれば、もっと正確にモデルを理解できるような気がします。

一番精度が高かったStock_Model_1がどれくらい正確かというと、下のtest dataの結果が参考になると思います。赤が本来の株価（closing price only）、青がLSTMが予測したt+1の株価（closing price only）です。

{{< figure src="/img/result.png" >}}

ざっとみると、t+1の値がtよりかけ離れていてもt-120までの値が安定しているならば、割と良い予測（t = 0~300までの部分）ができているように思います。しかしt > 300以降は、株価のスイングの幅が大きくなるとともに予測が乖離していっています。もしかしすると**このような連続的に大きなスイングをするデータが足りなかったために精度が落ちたのかもしれません。**より多くのデータを学習させることで精度はあがるかもしれません。

今後余力があれば行っていきたいこととしては、
- Company/ETF embeddingを足して、より多くのdataで学習する
- timeやpositionのembeddingをinputにconcatenateするよりも、複数のLSTM層に通してからモデルの途中にくっつける

などです。

## プロジェクトから学んだこと

最後に、自分がこのプロジェクトから学んだことをまとめていきます。一言でまとめるならば、

**Debugging neural networks is hard!!!**

ということです。

たった5日間の趣味程度のプロジェクトでしたが、非常に学ぶことは多かったです。このような解決すべき"ニーズ"を見つけた今、今度はDeep learning周りのtoolやframeworkを一通りみていこうと思います。