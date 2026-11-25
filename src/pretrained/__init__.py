"""Pre-trained (off-the-shelf) decision models, called as-is with no fine-tuning.

These test generalization: the same three-stage schema the fine-tuned models
were trained on, sent zero-shot to hosted decision models through the LiteLLM
proxy's System One route (/v1/systemone).
"""
