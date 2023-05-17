"""Converter for LLaMa checkpoints in the Hugging Face format."""

import argparse
import gc

import ctranslate2
import torch
import transformers
from ctranslate2.converters.transformers import ModelLoader, register_loader


@register_loader("ChatGLMConfig")
class ChatGLMLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "ChatGLMForConditionalGeneration"

    def get_model_spec(self, model):
        spec = ctranslate2.specs.TransformerDecoderModelSpec.from_config(
            model.config.num_layers,
            model.config.num_attention_heads,
            activation=ctranslate2.specs.Activation.GELU,
            pre_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
        )

        self.set_decoder(spec.decoder, model.transformer)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def get_vocabulary(self, model, tokenizer):
        embedding_size = model.transformer.word_embeddings.weight.shape[0]
        vocab_size = tokenizer.vocab_size
        vocab_token = [
            token
            for token, _ in sorted(
                tokenizer.get_vocab().items(), key=lambda item: item[1]
            )
        ]
        # hack here. for more details, see https://github.com/THUDM/ChatGLM-6B/issues/987
        extra_tokens = [tokenizer.unk_token for _ in range(embedding_size - vocab_size)]
        return vocab_token + extra_tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight.numpy()
        spec.beta = layer_norm.bias.numpy()

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.word_embeddings)
        self.set_layer_norm(spec.layer_norm, module.final_layernorm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            layer_spec.self_attention.linear[
                0
            ].weight = layer.attention.query_key_value.weight.numpy()
            layer_spec.self_attention.linear[
                1
            ].weight = layer.attention.dense.weight.numpy()

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.dense_h_to_4h)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.dense_4h_to_h)

            delattr(layer, "attention")
            delattr(layer, "mlp")
            gc.collect()


class ChatGLM6BConverter(ctranslate2.converters.TransformersConverter):
    def _load(self):
        with torch.no_grad():
            loader = ChatGLMLoader()

            kwargs = {}
            if self._load_as_float16:
                kwargs["torch_dtype"] = torch.float16
            if self._revision:
                kwargs["revision"] = self._revision
            if self._low_cpu_mem_usage:
                kwargs["low_cpu_mem_usage"] = self._low_cpu_mem_usage

            model = transformers.AutoModel.from_pretrained(
                "THUDM/chatglm-6b", trust_remote_code=True
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                "THUDM/chatglm-6b", trust_remote_code=True
            )

            spec = loader(model, tokenizer)

            if self._activation_scales:
                activation_scales = torch.load(
                    self._activation_scales, map_location="cpu"
                )
                loader.smooth_activation(spec, activation_scales)

            spec.register_file(tokenizer.sp_tokenizer.vocab_file)

            return spec


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ctranslate2.converters.Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = ChatGLM6BConverter(
        "THUDM/chatglm-6b",
        load_as_float16=True,
        low_cpu_mem_usage=True,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()
