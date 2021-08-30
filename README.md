# Vanilla transformer 

This is a bare-bones vanilla transformer as described in the 2017 paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The best way to learn a particular deep learning architecture is to work through implementing it yourself -- since transformers are so ubiquitous in modern deep learning, I went through the process of implementing one myself. 

## Model train

To train a model, just run `python train.py`. A reference German sentence is hard coded in the training script, and the English translation will display after every epoch. After 16 epochs, this the translation I got:

```
Epoch 16 mean loss: 0.8075555422668814
[2, 4, 644, 674, 2163, 232, 19, 11, 2809, 4478, 5, 3]
==== TRANSLATION ====
['<sos>', 'a', 'marathon', 'runner', 'jogs', 'past', 'people', 'and', 'portable', 'toilets', '.', '<eos>']
```

In Google translate, it comes out as:

```
a marathon runner runs past passers-by and mobile toilets
```

which is pretty close! Of course, this sentence was taken from the training set so the model (unexpectedly) learnt it quite quickly. **Soon I'll add the BLEU score metric to see how well it does**.

## Resources

There are several great resources for transformers, but these are the main ones which helped me understand the conceptual underpinnings quite a bit:

* Getting started with the [attention mechansim](http://peterbloem.nl/blog/transformers)
* [Positional encoder](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
* Lillian Weng's [transformer family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)
* Jay Alammar's [illustrated transformer](https://jalammar.github.io/illustrated-transformer/)

As for implementational details, I found [Aladdin Persson's](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py) reference implementation (without the original paper's positional encoder) to be very helpful, as well as his [seq2seq transformertutorial](https://www.youtube.com/watch?v=M6adRGJe5cQ).  

