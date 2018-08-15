# Codes usage:

## HW2-1: Caption Generation

### S2VT without any tips (Attention, Schedule Sampling, Beamsearch, ...etc).
1. Build word2id dict if `word2id.pkl` is not existed:
```
python3 parser_dict.py
```
2. Run `s2vt_v2.py`:

For training from scratch:
```
python3 s2vt_2.py -r 0
```

For resume training:
```
python3 s2vt_2.py -r [Lattest Epoch of already-saved model]
```

For testing:
```
python3 s2vt_v2.py --test -r [Lattest Epoch of already-saved model]
```

### S2VT with schedule sampling

Probability with linear decay: `prob = 1.0 - EPOCH*0.003`, `EPOCH = 1 ~ 200`.  
Achieve bleu score = 0.647 @175 epoch

