import os
import hashlib
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import bm25s


class HippDex:
    def __init__(self, model, embedder: Embedding, model_type: str = "GGUF") -> None:
        self.model = model
        self.type = model_type
        self.embedder = embedder
        self.indexer = bm25s.BM25()
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]
        self.corpus = []

    def generate(
        self,
        msg: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        repeat_penalty: float = 1.0,
    ):
        memories = []
        if self.embeddings is not None:
            memories = self.embedder.get_similar(msg)
        if self.corpus:
            tokens = bm25s.tokenize(msg)
            results, _ = self.indexer.retrieve(tokens, k=2)

            for i in range(results.shape[1]):
                index = results[0, i]
                memories.append(self.corpus[index])

        if len(memories) > 0:
            memories.insert(0, "[START OF OLD MEMORIES]")
            memories.insert(-1, "[END OF OLD MEMORIES]")
            msg += "\n"
            msg += "\n".join(memories)

        print(f"Memories:\n\n{memories}\n\n")

        self.history += [{"role": "user", "content": msg}]

        print(self.history)
        output = self.model.create_chat_completion(
            messages=self.history,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,
        )

        self.history += [output["choices"][0]["message"]]
        print(self.history)

        # self.embedder.embed({"user": msg, "assistant": output[""]})

        return output

    def save_state(self):
        pass

    def store(self):

        self.corpus = []
        for chat in self.history:
            self.corpus.append(chat["content"].split("[START OF OLD MEMORIES]")[0])
        tokens = bm25s.tokenize(self.corpus, stopwords="en")
        self.indexer.index(tokens)
        self.embedder.embed(self.corpus[-2])
        self.embedder.embed(self.corpus[-1])

    @property
    def embeddings(self):
        return self.embedder.embeddings

    @property
    def internal_state(self):
        return self.save_state()


class Embedding:
    def __init__(
        self,
        model: Optional[ort.InferenceSession] = None,
        model_path: Optional[str] = None,
        tokenizer=None,
    ):
        """
        model: onnxruntime.InferenceSession
        model_path: str

        Either pass a instance of a ONNX loaded using ort.InferenceSession and its tokenizer, or pass the path to the ONNX model directory.
        """

        if model:
            if not tokenizer:
                raise ValueError("Pass a tokenizer when directly passing the model")
            self.session = model
            self.tokenizer = tokenizer

        # Load tokenizer
        elif model_path:
            self.tokenizer = Tokenizer.from_file(
                os.path.join(model_path, "tokenizer.json")
            )
            self.session = ort.InferenceSession(
                os.path.join(model_path, "onnx", "model.onnx"),
                providers=["CPUExecutionProvider"],
            )
        else:
            raise ValueError("Atleast pass a model and its tokenizer, or model_path")

        self.tokenizer.enable_truncation(max_length=16)
        self.tokenizer.enable_padding(length=16)

        # Load ONNX model
        self.embeddings = None
        self.texts = {}

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling"""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, token_embeddings.shape
        ).astype(float)

        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    def _get_embeddings(self, text):
        encoded = self.tokenizer.encode(text)
        # Run inference
        outputs = self.session.run(
            None,
            {
                "input_ids": np.asarray([encoded.ids], dtype=np.int64),
                "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64),
                "token_type_ids": np.asarray([encoded.type_ids], dtype=np.int64),
            },
        )

        # Mean pooling
        embeddings = self._mean_pooling(outputs[0], encoded.attention_mask)

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings[0]

    def get_similar(self, text, sim_threshold=0.8) -> List:
        if self.embeddings is None:
            return []
        embedding = self._get_embeddings(text)

        dots = self.embeddings @ embedding
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(embedding)
        sim = dots / norms
        matched_idx = np.where(sim > sim_threshold)[0]
        matched_idx = matched_idx[np.argsort(sim[matched_idx])]
        results = [list(self.texts.values())[i] for i in matched_idx]

        return results

    def embed(self, text):
        """Encode text to embedding"""
        # Tokenize
        sha = hashlib.sha256(text.encode()).hexdigest()
        if sha in self.texts.keys():
            return

        embeddings = self._get_embeddings(text)

        if self.embeddings is not None:
            self.embeddings = np.append(
                self.embeddings, embeddings.reshape(1, -1), axis=0
            )
        else:
            self.embeddings = np.array([embeddings])

        self.texts[sha] = text
