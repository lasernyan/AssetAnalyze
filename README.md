# マルチアセット売買指標ツール

日米株・金利・金・資源・VIX・為替を一括監視し、画像の数式に基づいた売買指標を表示します。
**自動売買機能はありません。** あくまで売買判断の参考指標を提供するツールです。

## 対象アセット

| クラス | 銘柄 |
|--------|------|
| 日本株 | 日経225, TOPIX, トヨタ, ソニー, ソフトバンクG |
| 米国株 | S&P500, NASDAQ, ダウ, Apple, NVIDIA |
| 金利   | 米10年債, 米5年債, 米3ヶ月TB |
| 金     | 金先物(GC), 金ETF(GLD) |
| 資源   | WTI原油, 天然ガス, 銅先物, コモディティETF |
| VIX    | VIX恐怖指数, VXN(NASDAQ VIX) |
| 為替   | USD/JPY, EUR/JPY, EUR/USD, GBP/JPY |

## 使用する数式

### エッジ検出 (Edge Detection)
```
EV   = p · b - (1 - p)               期待値
edge = p_model - p_mkt               市場エッジ  (>0.04 で売買機会)
δ    = (p_model - p_mkt) / σ         ミスプライシングスコア
BS   = (1/n) · Σ(p_i - o_i)²        Brier Score (較正精度)
```

### ポジションサイジング (Position Sizing)
```
f*   = (p · b - q) / b               Kelly基準
f    = α · f*,  α = 0.25             1/4 Kelly (保守的)
VaR  = μ - 1.645 · σ                 VaR 95% (日次最大損失)
MDD  = (Peak - Trough) / Peak        最大ドローダウン (>8% で警告)
```

### パフォーマンス (Performance & Technical)
```
SR   = (E[R] - Rf) / σ(R)            Sharpe Ratio  (目標 > 2.0)
PF   = gross_profit / gross_loss     Profit Factor (目標 > 1.5)
RSI  (14日) < 30 = 売られ過ぎ / > 70 = 買われ過ぎ
MA   50日 vs 200日  Golden/Death Cross
MACD (12/26/9)  ヒストグラムの正負
Bollinger %B  0未満=下抜け / 1超=上抜け
```

### シグナルスコア
```
-100 ← SELL ← -30  HOLD  +30 → BUY → +100
```
各指標の重み合計。±30 以上でシグナル点灯。

## セットアップ

```bash
# 依存ライブラリのインストール
pip install multitasking==0.0.11   # yfinance の依存関係
pip install -r requirements.txt
```

## 使い方

```bash
# 全アセット (yfinance から実データ取得、失敗時はモックにフォールバック)
python main.py

# デモモード (ネット不要・モックデータ)
python main.py --demo

# 特定クラスのみ
python main.py --class 日本株
python main.py --class 米国株
python main.py --class 為替

# バンクロール・Kelly分数の変更
python main.py --bankroll 5000000 --alpha 0.5

# 直列実行 (並列ダウンロードを無効化)
python main.py --no-parallel
```

## ファイル構成

```
AssetAnalyze/
├── main.py                    # エントリポイント
├── config.py                  # アセット定義・パラメータ
├── requirements.txt
├── data/
│   ├── fetcher.py             # yfinanceラッパー (モックフォールバック付き)
│   └── mock.py                # GBMによる合成データ生成
├── indicators/
│   ├── edge_detection.py      # EV, Edge, Bayes, Brier, Mispricing δ
│   ├── position_sizing.py     # Kelly, VaR, MDD
│   ├── performance.py         # Sharpe, PF, RSI, MACD, Bollinger, ATR
│   └── analyzer.py            # 銘柄ごとの分析オーケストレーター
└── report/
    └── renderer.py            # Rich ターミナル表示
```

## 指標の説明

| 列名 | 説明 |
|------|------|
| 価格 | 最新終値 |
| 1日/5日/20日 | リターン |
| RSI | 14日RSI。緑<30(売られ過ぎ), 赤>70(買われ過ぎ) |
| MA trend | GOLDEN=ゴールデンクロス, DEATH=デッドクロス, NONE=なし |
| MACD | ▲=ヒストグラム正(上昇), ▼=負(下落) |
| BB %B | Bollinger %B。0未満=下抜け, 1超=上抜け |
| Edge | p_model − p_mkt。>0.04 で有意なエッジ |
| Mispricing δ | Z-score。|δ|>1.5 で顕著なミスプライシング |
| Kelly f | 推奨ポジションサイズ(バンクロール比) |
| VaR 95% | 日次最大損失の目安 (95%信頼) |
| MDD | 最大ドローダウン。赤字=8%超え警告 |
| Sharpe | 年率Sharpe Ratio |
| PF | Profit Factor |
| シグナル | BUY / SELL / HOLD |
| スコア | −100〜+100 の複合スコアバー |

> **免責事項**: 本ツールは情報提供のみを目的としており、投資助言ではありません。
> 実際の投資判断はご自身の責任で行ってください。
