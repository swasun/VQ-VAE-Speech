# Embedding stats

## Algorithm

- Evaluate the model using the val dataset. Save each resulting
embedding in a pickle file, with the corresponding speaker;
- Group the embeddings by speaker;
- Compute the distribution of each embedding (seaborn histogram, softmax);
- Compute all the distances between all possible distribution couples, using
a distribution distance (e.g. entropy) and plot them (seaborn histogram?).

## What should we noticed:

-> Check if the entropy result is close to the uniform distribution with
and without speaker embedding (do all the previous steps using speaker
embedding as well);
-> Because 29 codebook vectors compressed as shit the data, it's likely
already speaker independent. We can check thazt by increasing the number
of embedding vectors (e.g. the higher, finess the representation will be).

# Embedding stats 2

## Idea

Increase exponentially the number of embedding vectors and notice if the mapping accuracy increase, at least linearly (it should be the case)

# Phonemes stats

Goal: Check if each vector in the codebook correspond to a specific phoneme.

## Algorithm

- Make a 2D projection of the embedding vectors using umap;
- Plot the projection result as points, and the embedding indices as marks;
- Map each groundtruth phoneme with each mark (how?).
