import datasets

# https://huggingface.co/datasets/hotpotqa/hotpot_qa?row=16
# https://arxiv.org/pdf/1606.05250
dataset = datasets.load_dataset("rajpurkar/squad")

from collections import Counter
import matplotlib.pyplot as plt

# Get all characters from all IDs
all_chars = ''.join(dataset['train']['id'] + dataset['validation']['id'])
char_counts = Counter(all_chars)

# Create lists for characters and their counts
chars = list(char_counts.keys())
counts = list(char_counts.values())

# Create histogram
plt.figure(figsize=(12, 6))
plt.bar(chars, counts)
plt.title('Character Distribution in SQuAD IDs')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("squad_ids.png")