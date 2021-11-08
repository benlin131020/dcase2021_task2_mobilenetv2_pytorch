for i in {1..7};
do
    python 00_train.py -d;
done
python emb.py -d
python ML.py
python 01_test_ml.py -d