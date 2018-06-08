使用三种方式实现ner
1. lstm
3. bi_lstm
4. bi_lstm + crf
效果：
lstm < bi_lstm < bi_lstm+crf

另：用crf++ 训练
注：
  crf++ 训练速度非常快，而且效果也很不错，介于bi_lstm 与bi_lstm_crf 之间
