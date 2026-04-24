import os, json
from typing import Any, Optional
import numpy as np
from llama_cpp.llama import Llama
import onnxruntime as ort
from tokenizers import Tokenizer


class HippDex:
    def __init__(self, model, model_type: str) -> None:
        self.model = model
        self.type = model_type
        self.embedder = Embedding()

    def generate(self, msg: str):
        pass

    def save_state(self):
        pass

    @property
    def embeddings(self):
        pass

    @property
    def internal_state(self):
        return self.save_state()


class Embedding:
    def __init__(
        self,
        model: Optional[ort.InferenceSession] = None,
        model_path: Optional[str] = None,
        tokenizer=None,
    ) -> None:
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
        self.embeddings = np.array()

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling"""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(
            input_mask_expanded, token_embeddings.shape
        ).astype(float)

        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    def embed(self, text):
        """Encode text to embedding"""
        # Tokenize
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

        self.embeddings = np.append(self.embeddings, embeddings[0])
