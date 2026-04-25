import os
import hashlib
from typing import List, Optional
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class HippDex:
    def __init__(self, model, model_type: str) -> None:
        self.model = model
        self.type = model_type
        self.embedder = Embedding()

    def generate(
        self,
        msg: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        repeat_penaly: float = 1.0,
    ):
        if self.embeddings is not None:
            memories = "\n".join(self.embedder.get_similar(msg))
            msg += memories

        output = self.model.generate(
            msg,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penaly=repeat_penaly,
        )

        return output

    def save_state(self):
        pass

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

        return embeddings[0].

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
            self.embeddings = np.append(self.embeddings, embeddings.reshape(1, -1), axis=0)
        else:
            self.embeddings = np.array([embeddings])

        self.texts[sha] = text
