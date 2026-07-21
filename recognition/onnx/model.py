from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ap_res_models_inference import get_model


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


# ----------------------------------------------------------- the service Model
class Model:
    '''
    Class for recognizing handwritten text on line images (TrOCR over ONNX).

    The model is split into three ONNX graphs, the autoregressive generation loop
    lives here in numpy (no torch / transformers at serving time):
        encoder            - pixel_values -> encoder_hidden_states, runs once per line
        decoder            - first decoder step: emits logits + the KV cache ("present.*")
        decoder_with_past  - one token per call: last token + past KV -> logits + updated KV
    '''

    def __init__(self, models_cfg):
        p = models_cfg['parametrs']

        # Preprocessing (must match the training image processor)
        self.image_size = int(p['image_size'])
        self.image_mean = np.asarray(p['image_mean'], dtype=np.float32)
        self.image_std = np.asarray(p['image_std'], dtype=np.float32)
        self.resample = int(p['resample'])           # PIL code: 2 - bilinear, 3 - bicubic

        # Constants
        self.bos_token_id = int(p['bos_token_id'])   # decoder_start_token_id
        self.eos_token_id = int(p['eos_token_id'])
        self.pad_token_id = int(p['pad_token_id'])
        self.max_length = int(p['max_length'])
        self.num_beams = int(p.get('num_beams', 1))
        self.length_penalty = float(p.get('length_penalty', 1.0))

        self.tokenizer = Tokenizer.from_file(p['tokenizer_path'])

        self.models = {}
        self._load_models(models_cfg)

    def _load_models(self, models_cfg):
        """Load all ONNX models from config using get_model."""
        for model_name in ['encoder', 'decoder', 'decoder_with_past']:
            cfg = models_cfg[model_name]
            self.models[model_name] = get_model(
                engine_name=cfg['engine']['engine_name'],
                path_to_weights=cfg['engine']['weights_path'],
                use_gpu=cfg['engine']['use_gpu'],
            )
            logging.info(f"Loaded {model_name} from {cfg['engine']['weights_path']}")

    # ------------------------------------------------------------ preprocessing / encoder
    def preprocess(self, imgs):
        """list of PIL RGB -> np.ndarray float32 [N, 3, S, S]."""
        out = []
        for img in imgs:
            img = img.convert('RGB').resize((self.image_size, self.image_size), self.resample)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            out.append(((arr - self.image_mean) / self.image_std).transpose(2, 0, 1))
        return np.stack(out)

    def encode(self, imgs):
        """list of PIL RGB -> encoder_hidden_states np.ndarray [N, enc_seq, hidden]."""
        return self.models['encoder'](pixel_values=self.preprocess(imgs))

    # ------------------------------------------------------------ decoder steps
    def _first_step(self, ids, encoder_hidden_states):
        """First decoder step: returns logits and the full KV cache keyed 'present.*'."""
        outs = self.models['decoder'](**{
            'input_ids': ids,
            'encoder_hidden_states': encoder_hidden_states,
        })
        logits = outs['logits']
        cache = {name: value for name, value in outs.items() if name.startswith('present')}
        return logits, cache

    def _next_step(self, last_ids, cache):
        """One generation step: the cache is fed back as 'past_key_values.*'."""
        feeds = {'input_ids': last_ids}
        for name, value in cache.items():
            feeds[name.replace('present', 'past_key_values', 1)] = value
        outs = self.models['decoder_with_past'](**feeds)
        for name, value in outs.items():
            if name.startswith('present'):
                cache[name] = value
        return outs['logits'], cache

    # ------------------------------------------------------------ generation
    def recognize(self, img, num_beams=None, max_length=None):
        """Recognize the text of a single line image (the service contract)."""
        return self.recognize_batch([img], num_beams=num_beams, max_length=max_length)[0]

    def recognize_batch(self, imgs, num_beams=None, max_length=None):
        """Recognize a batch of line images. Preprocessing + encoder run once for the
        whole batch; num_beams == 1 -> one batched greedy loop, num_beams > 1 -> beam
        search per line over the shared encoder states."""
        num_beams = int(num_beams or self.num_beams)
        max_len = int(max_length or self.max_length)
        encoder_hidden_states = self.encode(imgs)

        if num_beams <= 1:
            seqs = self._greedy(encoder_hidden_states, max_len)
        else:
            seqs = [self._beam(encoder_hidden_states[i:i + 1], num_beams, max_len)
                    for i in range(len(imgs))]
        return [self.tokenizer.decode(s, skip_special_tokens=True).strip() for s in seqs]

    def _greedy(self, encoder_hidden_states, max_len):
        """Batched greedy loop: argmax token per row per step, rows stop at EOS."""
        batch = encoder_hidden_states.shape[0]
        ids = np.full((batch, 1), self.bos_token_id, dtype=np.int64)
        logits, cache = self._first_step(ids, encoder_hidden_states)

        seqs = [[] for _ in range(batch)]
        finished = np.zeros(batch, dtype=bool)
        for _ in range(max_len - 1):
            next_token = logits[:, -1, :].argmax(-1)
            next_token = np.where(finished, self.pad_token_id, next_token)
            for idx in range(batch):
                if not finished[idx]:
                    if next_token[idx] == self.eos_token_id:
                        finished[idx] = True
                    else:
                        seqs[idx].append(int(next_token[idx]))
            if finished.all():
                break
            logits, cache = self._next_step(
                next_token.reshape(batch, 1).astype(np.int64), cache)
        return seqs

    def _beam(self, encoder_hidden_states, num_beams, max_len):
        """Beam search for one line ([1, ...] encoder states). Returns best token ids."""
        logits, cache = self._first_step(
            np.array([[self.bos_token_id]], dtype=np.int64), encoder_hidden_states)
        log_probs = _log_softmax(logits[0, -1, :].astype(np.float64))
        log_probs[self.eos_token_id] = -np.inf          # no empty hypothesis
        top = np.argsort(log_probs)[::-1][:num_beams]
        scores = log_probs[top]
        seqs = [[int(t)] for t in top]
        cache = {k: np.repeat(v, num_beams, axis=0) for k, v in cache.items()}

        finished = []                                    # (normalized score, tokens)
        last = np.array(top, dtype=np.int64).reshape(-1, 1)
        for step in range(2, max_len):
            logits, cache = self._next_step(last, cache)
            log_probs = _log_softmax(logits[:, -1, :].astype(np.float64))   # (beams, V)
            vocab = log_probs.shape[-1]
            total = (scores[:, None] + log_probs).ravel()
            order = np.argsort(total)[::-1][: 2 * num_beams]

            new_seqs, new_scores, src_idx, next_tok = [], [], [], []
            for cand in order:
                beam, token = int(cand // vocab), int(cand % vocab)
                if token == self.eos_token_id:
                    finished.append((total[cand] / (step ** self.length_penalty), seqs[beam]))
                else:
                    new_seqs.append(seqs[beam] + [token])
                    new_scores.append(total[cand])
                    src_idx.append(beam)
                    next_tok.append(token)
                if len(new_seqs) == num_beams:
                    break
            if not new_seqs:
                break
            idx = np.array(src_idx)
            cache = {k: v[idx] for k, v in cache.items()}   # KV rows follow surviving beams
            seqs, scores = new_seqs, np.array(new_scores)
            last = np.array(next_tok, dtype=np.int64).reshape(-1, 1)

            if len(finished) >= num_beams:
                best_running = scores.max() / (max(step, 1) ** self.length_penalty)
                if best_running <= min(f[0] for f in finished[:num_beams]):
                    break
        if not finished:                                    # ran out of length
            finished = [(s / (len(q) ** self.length_penalty), q)
                        for s, q in zip(scores, seqs)]
        return max(finished, key=lambda f: f[0])[1]


# ----------------------------------------------------------- experiments helper
def load_model(onnx_dir, num_beams=None, use_gpu=0):
    """Build the service-style models_cfg from an export folder (service_config.json)."""
    d = Path(onnx_dir)
    svc = json.loads((d / "service_config.json").read_text(encoding="utf-8"))

    def engine(name):
        return {"engine": {"weights_path": str(d / f"{name}.onnx"),
                           "engine_name": "onnx", "use_gpu": use_gpu}}

    models_cfg = {
        "encoder": engine("encoder_model"),
        "decoder": engine("decoder_model"),
        "decoder_with_past": engine("decoder_with_past_model"),
        "parametrs": {
            "image_size": svc["image_size"],
            "image_mean": svc["image_mean"],
            "image_std": svc["image_std"],
            "resample": svc["resample"],
            "bos_token_id": svc["decoder_start_token_id"],
            "eos_token_id": svc["eos_token_id"],
            "pad_token_id": svc["pad_token_id"],
            "max_length": svc["max_length"],
            "num_beams": num_beams if num_beams is not None else 4,
            "tokenizer_path": str(d / "tokenizer.json"),
        },
    }
    return Model(models_cfg)
