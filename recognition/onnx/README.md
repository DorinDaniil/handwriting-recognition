# ONNX export & light serving runtime

Wraps the TrOCR recognizer for a service **without torch/transformers**. The model is split
into three graphs — that's the standard way to keep the autoregressive step fast:

| file | role |
|---|---|
| `encoder_model.onnx` | `pixel_values -> encoder_hidden_states`, runs **once** per crop |
| `decoder_model.onnx` | first decoder step; also emits the KV cache |
| `decoder_with_past_model.onnx` | one token per call, consumes/updates the KV cache |

The generation loop (greedy / beam) is plain numpy in `runtime_onnx.py`. The tokenizer is the
byte-level BPE saved as `tokenizer.json`, loaded by the light `tokenizers` lib. Preprocessing
(resize + 0.5/0.5 normalize) is PIL+numpy, parameters recorded in `service_config.json`.

## 1. Export (on the training box — needs torch, transformers, optimum)

```bash
pip install optimum[exporters]
python onnx/export_onnx.py --ckpt outputs/trocr_small_bi_finetune_with_hwr200_cleaned/best
python onnx/export_onnx.py --ckpt ... --quantize      # extra *_int8.onnx for CPU serving
```

## 2. Verify parity vs torch (same box)

Open **`compare_onnx.ipynb`**: it spells out the generation loop (batched greedy + beam with
KV-cache reorder) over the three graphs, runs the torch reference on the same crops, and prints
exact-match rate / char similarity / latency plus visual samples. Greedy should match torch
~exactly; beam-4 judge by char similarity (ties may break differently). Re-create `rec` with
`int8=True` after quantizing — check the similarity stays acceptable.

## 3. Serve (copy to the service)

Copy `runtime_onnx.py` + the export folder (3 `.onnx`, `tokenizer.json`, `service_config.json`);
deps in `requirements_service.txt` (onnxruntime, tokenizers, numpy, Pillow).

```python
from runtime_onnx import OnnxTrOCR

rec = OnnxTrOCR("onnx_out")                       # int8=True for the quantized graphs
texts = rec.recognize(pil_crops)                  # greedy, batched — fastest
text = rec.recognize_beam(crop, num_beams=4)      # eval-quality decoding
```

Notes:
- GPU serving: `pip install onnxruntime-gpu`, then `OnnxTrOCR(..., providers=["CUDAExecutionProvider", "CPUExecutionProvider"])`.
- Greedy is ~num_beams× cheaper than beam; measure whether beam-4 actually buys CER on your data before shipping it.
- One monolithic ONNX with the generate-loop inside is possible (ONNX control-flow ops) but painful and slower to iterate on — the 3-graph split + outer loop is the standard used by optimum/ORT.
