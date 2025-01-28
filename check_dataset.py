from datasets import load_dataset
import matplotlib.pyplot as plt
from icecream import ic

raw_datasets = load_dataset("bookcorpus", split="train[:5%]")

sentence_lengths = [len(sentence.split()) for sentence in raw_datasets['text']]

plt.figure(figsize=(10, 6))
plt.hist(sentence_lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')

plt.savefig('sentence_length_distribution.png', format='png', dpi=300)

plt.show()