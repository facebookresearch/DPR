from models.biencoder import BiEncoder
from models.fairseq_models import Wav2Vec2Encoder
from models.hf_models import HFBertEncoder


def get_audio_mixed_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0

    question_encoder = Wav2Vec2Encoder(
        cfg.wav2vec_cp_file, cfg.wav2vec_apply_mask, cfg.max_audio_t
    )

    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    fix_ctx_encoder = cfg.fix_ctx_encoder if hasattr(cfg, "fix_ctx_encoder") else False

    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer
