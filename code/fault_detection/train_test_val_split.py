"""
How to do this train test val split

Need one for each category

Then try one for all categories combined.

train test validation split -> 0.7, 0.1, 0.2

How do we do a proper split, because the pairs are random... 
You have to choose somehow, But should we save the pairs so it is consistent over everything?

Probably easiest thing for training is to keep a set of indices

Then for validation and test, we need the indices, and the pairs and targets themselves, as they are randomly generated.

but for the validation and testing, we need the same pairs -> so save a set, and then for testing it can be different
So set a seed -> Use default rng for both the index generator and the dataset


TO TEST

Binary baseline for all cats 
- pretrained
- non pretrained

Multiclass baseline
- pretrained
- non pretrained

View combination
- pretrained
- non pretrained
"""

