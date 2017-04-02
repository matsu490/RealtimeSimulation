# Change Log
## [v0.2.0](https://github.com/matsu490/RealtimeSimulation/tree/v0.2.0) (2017--)
**Implemented enhancements:**
- scipy.integrate.odeint() を用いた数値積分が使えるようになった
- LIF (leaky integrate-and-fire) モデルを実装した
- HH (Hodgkin-Huxley) モデルを実装した
- ガウシアンノイズの入力を実装した
- 積分を必要としない入力のためのクラス（Generator）を実装した

## [v0.1.0](https://github.com/matsu490/RealtimeSimulation/tree/v0.1.0) (2017-2-17)
**Implemented enhancements:**
- 初めてのリリース
- 単一細胞シミュレーションが可能
- 今のところ Acker's モデルのみ実装

## ToDo
- [ ] ネットワークシミュレーション
- [ ] レコードボタンの追加とデータ保存機能の追加する
- [ ] 一定間隔で外部刺激を与えるための UI を追加する
- [x] LIF モデルを追加する
- [x] ノイズを入れる
- [ ] ポアソンスパイク生成器を作る
- [ ] DC などの外部入力を Generator クラスなどとしてまとめる
