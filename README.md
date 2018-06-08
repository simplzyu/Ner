使用三种方式实现ner
1. lstm
3. bi_lstm
4. bi_lstm + crf
效果：
lstm < bi_lstm < bi_lstm+crf

另：用crf++ 训练
注：
  crf++ 训练速度非常快，而且效果也很不错，介于bi_lstm 与bi_lstm_crf 之间
    O B M E的f1_score: [0.96304013 0.95219255 0.9702511  0.944939  ]; accuracy 0.9645588356220706 ;avg_f1_score 0.9576056979336123
  训练结果在./log 文件夹中
