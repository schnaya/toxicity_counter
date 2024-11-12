import json
from configparser import ConfigParser

import requests
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import MarianMTModel, MarianTokenizer, M2M100Tokenizer, M2M100ForConditionalGeneration
ACCESS_TOKEN = "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"
url = "https://huggingface.co/None/resolve/main/config.json"

class TranslationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config_path=None):
        if config_path:
            config = ConfigParser()
            config.read(config_path)
            self.lang_to_id = {lang: int(id) for lang, id in config.items("languages")}
        self.lang_to_model = {
            "ru": "Helsinki-NLP/opus-mt-ru-en",
            "es": "Helsinki-NLP/opus-mt-es-en",
            "fr": "Helsinki-NLP/opus-mt-fr-en",
            "it": "Helsinki-NLP/opus-mt-it-en",
            "tr": "Helsinki-NLP/opus-mt-tr-en",
            "pt": "facebook/m2m100_418M"
        }

        self.models = {}
        self.tokenizers = {}
        for lang, lang_id in self.lang_to_id.items():
            if lang == "en":
                continue
            model_name = self.lang_to_model.get(lang)
            if model_name == "facebook/m2m100_418M":
                model = M2M100ForConditionalGeneration.from_pretrained(model_name)
                tokenizer = M2M100Tokenizer.from_pretrained(model_name)
                tokenizer.src_lang = "pt"  # Для португальского языка
            else:
                model = MarianMTModel.from_pretrained(model_name)
                tokenizer = MarianTokenizer.from_pretrained(model_name)

            self.models[lang] = model
            self.tokenizers[lang] = tokenizer

    def translate(self, text, lang):

        if lang == "en":
            return text
        model = self.models.get(lang)
        tokenizer = self.tokenizers.get(lang)

        if not model or not tokenizer:
            raise ValueError(f"Translation model for language '{lang}' is not supported.")
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        if isinstance(model, M2M100ForConditionalGeneration):
            tokenizer.src_lang = lang
            model.config.forced_bos_token_id = tokenizer.get_lang_id("en")

        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text[0]

    def transform(self, texts, language_ids):
        translated_texts = []
        for text, lang_id in zip(texts, language_ids):
            lang_code = next((lang for lang, id_ in self.lang_to_id.items() if id_ == lang_id), None)
            if len(text) > 512:
                text = text[:512]
            translated_text = self.translate(text, lang_code)
            translated_texts.append(translated_text)
        return translated_texts