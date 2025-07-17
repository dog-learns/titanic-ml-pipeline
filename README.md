# Titanic ML Pipeline: From Preprocessing to Model Evaluation  
タイタニック機械学習パイプライン：前処理からモデル評価まで

This repository contains an end-to-end machine learning pipeline using the Titanic dataset from Kaggle.  
It includes data preprocessing, feature engineering, model training, hyperparameter tuning, ensembling, and SHAP-based interpretation.

このリポジトリでは、KaggleのTitanicデータセットを用いて、機械学習の一連の流れ（前処理〜モデル評価）を実装しています。  
特徴量エンジニアリングやハイパーパラメータのチューニング、アンサンブル、SHAPによるモデルの解釈までを含んでいます。

---

## 🎯 Objective / 目的

- Predict survival of passengers aboard the Titanic  
- タイタニック号の乗客の生存可否を予測する

---

## 🛠️ Techniques Used / 使用した技術

- Data cleaning & feature engineering / データの前処理と特徴量エンジニアリング  
- Models: RandomForest, XGBoost, LightGBM, Calibrated SVM  
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)  
- Stacking ensemble with XGBoost as meta learner / メタモデルにXGBoostを用いたスタッキング  
- SHAP (SHapley Additive exPlanations) for interpretability / SHAPによるモデル解釈

---

## 📈 Results / 結果

- Accuracy (CV): **83.6%**  
- ROC AUC (CV): **87.4%**  
- SHAP analysis shows strongest contributions from XGBoost and RandomForest  
- SHAP解析では、XGBoostとRandomForestが最も大きな貢献をしていることが分かりました

---

## 📁 Files / ファイル構成

- `Taitanic_ml_pipline.ipynb`  
　Main notebook with code and bilingual (EN/JP) explanation  
　コードと英語＋日本語の解説付きノートブック

---

## 💻 How to Use / 使い方

1. Download the Titanic dataset (`train.csv`, `test.csv`) from [Kaggle](https://www.kaggle.com/c/titanic)  
　KaggleのTitanicコンペページから `train.csv` と `test.csv` をダウンロードします

2. Upload them to your Google Drive  
　それらのファイルをGoogleドライブにアップロードします

3. Open `Taitanic_notebook.ipynb` in [Google Colab](https://colab.research.google.com/)  
　Google Colabで `Taitanic_notebook.ipynb` を開きます

4. Run the notebook step-by-step  
　一つずつセルを実行していきます

※ The notebook is designed to run on Google Colab  
※ ノートブックはGoogle Colab上で動作することを前提としています

---

## 📚 Intended Audience / 想定読者

- Beginners to intermediate learners who want to understand a full ML workflow  
　機械学習の流れ（前処理から評価まで）を学びたい初学者〜中級者

- Those creating a portfolio for career change or job hunting  
　転職や就職活動に向けてポートフォリオを作成中の方

- Anyone interested in practical examples of SHAP or ensemble learning  
　SHAPやアンサンブル学習の実践例を探している人

---

## 🧑‍💻 Author / 作成者

GitHub: [dog-learns]  
Feel free to star ⭐ this repo or leave feedback!  
お気軽にStarやご意見をお寄せください😊
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dog-learns/titanic-ml-pipeline/blob/main/titanic_ml_pipeline.ipynb)
