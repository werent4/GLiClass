from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.clap.configuration_clap import ClapTextConfig, ClapAudioConfig
from transformers.models.auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

CONFIG_MAPPING['clap_text_model'] = ClapTextConfig
CONFIG_MAPPING['clap_audio_model'] = ClapAudioConfig


class GLiClassModelConfig(PretrainedConfig):
    model_type = "GLiClass"
    is_composition = True

    def __init__(
        self,
        # Encoder configs (existing)
        encoder_config=None,
        encoder_model=None,
        label_model_config=None,
        label_model_name=None,
        audio_model_config=None,
        audio_model_name=None,
        
        # Token indices
        class_token_index=-1,
        text_token_index=-1,
        audio_token_index=-1,
        ignore_index=-100,
        
        # Model dimensions
        hidden_size=None,
        projector_hidden_act="gelu",
        projection_dim=512,
        vocab_size=None,
        
        # Task config
        problem_type='single_label_classification',
        max_num_classes=25,
        use_lstm=False,
        initializer_range=0.03,
        scorer_type='simple',
        pooling_strategy='first',
        
        # Focal loss
        focal_loss_alpha=0.5,
        focal_loss_gamma=2,
        contrastive_loss_coef=0,
        audio_text_contrastive_coef=0.5,
        audio_text_temperature=0.07,
        learnable_temperature=False,
        
        # Features
        logit_scale_init_value=2.6592,
        normalize_features=False,
        extract_text_features=False,
        architecture_type='uni-encoder',
        prompt_first=False,
        squeeze_layers=False,
        embed_class_token=True,
        init_from_larger_clap=False,
        
        # PEAudioVisual specific
        pe_model_name="pe-av-base",
        pe_pretrained=True,
        hidden_dropout_prob=0.1,
        
        **kwargs,
    ):
        # === Existing encoder config logic ===
        if isinstance(encoder_config, dict):
            encoder_config["model_type"] = (
                encoder_config["model_type"]
                if "model_type" in encoder_config
                else "deberta-v2"
            )
            if encoder_config["model_type"] == "clap_text_model":
                encoder_config = ClapTextConfig(**encoder_config)
            else:
                encoder_config = CONFIG_MAPPING[encoder_config["model_type"]](**encoder_config)
        elif encoder_config is None:
            encoder_config = CONFIG_MAPPING["deberta-v2"]()

        self.encoder_config = encoder_config
        self.encoder_model_name = encoder_model

        if label_model_name is not None:
            if isinstance(label_model_config, dict):
                label_model_config["model_type"] = (
                    label_model_config["model_type"]
                    if "model_type" in label_model_config
                    else "deberta-v2"
                )
                label_model_config = CONFIG_MAPPING[label_model_config["model_type"]](**label_model_config)
            elif label_model_config is None:
                label_model_config = CONFIG_MAPPING["deberta-v2"]()

            self.label_model_config = label_model_config
        else:
            self.label_model_config = None
        self.label_model_name = label_model_name

        if audio_model_name is not None:
            if isinstance(audio_model_config, dict):
                audio_model_config["model_type"] = (
                    audio_model_config["model_type"]
                    if "model_type" in audio_model_config
                    else "wav2vec2"
                )
                if audio_model_config["model_type"] == "clap_audio_model":
                    audio_model_config = ClapAudioConfig(**audio_model_config)
                else:
                    audio_model_config = CONFIG_MAPPING[audio_model_config["model_type"]](**audio_model_config)
            elif audio_model_config is None:
                audio_model_config = CONFIG_MAPPING["wav2vec2"]()

            self.audio_model_config = audio_model_config
        else:
            self.audio_model_config = None
        self.audio_model_name = audio_model_name

        if hidden_size is None:
            if architecture_type == 'audio-encoder' and audio_model_name is None:
                self.hidden_size = 768  # default for pe-av-base
            else:
                self.hidden_size = self.encoder_config.hidden_size
        else:
            self.hidden_size = hidden_size

        if vocab_size is None:
            if architecture_type == 'audio-encoder' and audio_model_name is None:
                self.vocab_size = 32000  # placeholder
            else:
                self.vocab_size = self.encoder_config.vocab_size
        else:
            self.vocab_size = vocab_size

        if class_token_index == -1:
            self.class_token_index = self.vocab_size
        else:
            self.class_token_index = class_token_index

        if text_token_index == -1:
            self.text_token_index = self.vocab_size + 1
        else:
            self.text_token_index = text_token_index

        if audio_token_index == -1:
            self.audio_token_index = self.vocab_size + 2
        else:
            self.audio_token_index = audio_token_index

        if architecture_type not in {"audio-bi-encoder"} and init_from_larger_clap:
            raise ValueError(
                f"Cannot init {architecture_type} projection layers from clap. "
                "Please ensure you are using audio based arch"
            )

        self.init_from_larger_clap = init_from_larger_clap
        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.projection_dim = projection_dim
        self.problem_type = problem_type
        self.max_num_classes = max_num_classes
        self.initializer_range = initializer_range
        self.scorer_type = scorer_type
        self.pooling_strategy = pooling_strategy
        self.use_lstm = use_lstm
        
        # Focal loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        
        self.contrastive_loss_coef = contrastive_loss_coef
        
        self.audio_text_contrastive_coef = audio_text_contrastive_coef
        self.audio_text_temperature = audio_text_temperature
        
        # Features
        self.logit_scale_init_value = logit_scale_init_value
        self.normalize_features = normalize_features
        self.extract_text_features = extract_text_features
        self.architecture_type = architecture_type
        self.prompt_first = prompt_first
        self.squeeze_layers = squeeze_layers
        self.embed_class_token = embed_class_token
        
        self.pe_model_name = pe_model_name
        self.pe_pretrained = pe_pretrained
        self.hidden_dropout_prob = hidden_dropout_prob

        super().__init__(**kwargs)