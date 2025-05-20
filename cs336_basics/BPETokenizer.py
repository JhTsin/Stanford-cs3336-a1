import os
import regex as re
from typing import BinaryIO
from collections import defaultdict
from multiprocessing import Process, Queue

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split on the special tokens"""
    parts = re.split("|".join(re.escape(tok) for tok in special_tokens), text)
    return parts

def pretokenize(text: str, special_tokens: list[str]) -> list[bytes]:
    parts = remove_special_tokens(text, special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_list = []
    for part in parts:
        str_tokens = re.findall(PAT, part)
        part_tokens = [s.encode('utf-8') for s in str_tokens]
        tokens_list.append(part_tokens)
    tokens = [token for part_tokens in tokens_list for token in part_tokens]
    return tokens

def worker(text: str, special_tokens: list[str], q: Queue):
    pretokens = pretokenize(text, special_tokens)
    q.put(pretokens)
    # print("done")

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_merges = max(vocab_size - len(special_tokens) - 256, 0)

    # Initialize vocab
    vocab = {}
    vocab = {x:bytes([x]) for x in range(0,256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")
    merges = []


    # Chunk the text file
    num_processes = 1
    chunk_list = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_list.append(chunk)

    # Parallelizing pretokenization
    pretokens_list = []
    processes = []
    q = Queue()
    for chunk in chunk_list:
        p = Process(target=worker, args=(chunk, special_tokens, q))
        p.start()
        processes.append(p)

    pretokens_list = [q.get() for _ in processes]

    for p in processes:
        p.join()
    # print("Pretokenization done")

    pretokens = [token for tokens in pretokens_list for token in tokens]

    # Merging
    for i in range(num_merges):
        counts = defaultdict(int)

        for pretoken in pretokens:
            for index1, index2 in zip(pretoken, pretoken[1:]):
                counts[index1, index2] += 1
        
        max_pair = max(counts, key=counts.get)
        index1, index2 = max_pair

        new_index = 256 + len(special_tokens) + i

        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))
        
        pretokens = merge(pretokens, max_pair, new_index)

    return (vocab, merges)

def merge(indices: list[list[int]], pair: (int, int), new_index: int) -> list[int]:
    """Merge the pairs with highest frequency"""
    new_indices = []
    i = 0

    while i < len(indices):
        j = 0
        pretoken = indices[i]
        new_pretoken = []

        while j < len(pretoken):
            if (j < len(pretoken)-1) and ((pretoken[j], pretoken[j+1]) == pair):
                new_pretoken.append(new_index)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1

        new_indices.append(new_pretoken)
        i += 1

    return new_indices


if __name__ == "__main__":
    file_path = "./data/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)

    # print({k : v for k,v in vocab.items() if k > 255})
    print(vocab)
    print(merges)