@echo off
echo ========================================================
echo Starting Automated Training Pipeline (20 Total Runs)
echo ========================================================

echo.
echo --------------------------------------------------------
echo SAMBA - 26 Features (Standard Indicators)
echo --------------------------------------------------------
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 1
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 2
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 3
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 4
python main.py --model samba --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 5

echo.
echo --------------------------------------------------------
echo SAMBA - 27 Features (With LLM Sentiment)
echo --------------------------------------------------------
python main.py --model samba --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 1
python main.py --model samba --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 2
python main.py --model samba --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 3
python main.py --model samba --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 4
python main.py --model samba --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 5

echo.
echo --------------------------------------------------------
echo LSTM - 26 Features (Standard Indicators)
echo --------------------------------------------------------
python main.py --model lstm --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 1
python main.py --model lstm --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 2
python main.py --model lstm --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 3
python main.py --model lstm --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 4
python main.py --model lstm --dataset "Dataset/sp500_with_indicators.csv" --num_features 26 --seed 5

echo.
echo --------------------------------------------------------
echo LSTM - 27 Features (With LLM Sentiment)
echo --------------------------------------------------------
python main.py --model lstm --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 1
python main.py --model lstm --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 2
python main.py --model lstm --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 3
python main.py --model lstm --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 4
python main.py --model lstm --dataset "Dataset/sp500_with_indicators_llm.csv" --num_features 27 --seed 5

echo.
echo ========================================================
echo All 20 training runs have completed successfully!
echo ========================================================
pause