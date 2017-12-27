# Transformer-tensorflow

Yet another tensorflow implemntation of [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

<img src="https://raw.githubusercontent.com/flrngel/Transformer-tensorflow/master/resources/transformer.jpg" width=300>

## Usage

First you need to download [IWSLT 2016 German–English parallel corpus](https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz) dataset as below

```
./data/IWSLT16
├── IWSLT16.TED.dev2010.de-en.de.xml
├── IWSLT16.TED.dev2010.de-en.en.xml
├── IWSLT16.TED.tst2010.de-en.de.xml
├── IWSLT16.TED.tst2010.de-en.en.xml
├── IWSLT16.TED.tst2011.de-en.de.xml
├── IWSLT16.TED.tst2011.de-en.en.xml
├── IWSLT16.TED.tst2012.de-en.de.xml
├── IWSLT16.TED.tst2012.de-en.en.xml
├── IWSLT16.TED.tst2013.de-en.de.xml
├── IWSLT16.TED.tst2013.de-en.en.xml
├── IWSLT16.TED.tst2014.de-en.de.xml
├── IWSLT16.TED.tst2014.de-en.en.xml
├── README
├── train.tags.de-en.de
└── train.tags.de-en.en
```

And then,
```bash
# Pre-process data
$ python preprocess.py --dataset IWSLT16
# Train IWSLT16
$ python train.py --dataset IWSLT16
```

## TO-DO

- BLEU Score (priority: high)
- Test Accuracy (priority: high)
- update learning rate as paper (priority: low)

## Training Loss

<img src="https://raw.githubusercontent.com/flrngel/Transformer-tensorflow/master/resources/loss.jpg" width=300>

## References

- [Kyubyong's transformer](https://github.com/Kyubyong/transformer)
- [DongjunLee's transformer](https://github.com/DongjunLee/transformer-tensorflow)
