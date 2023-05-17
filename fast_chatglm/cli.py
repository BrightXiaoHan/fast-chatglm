"""This basic script demonstrates how to progressively write the generation output.
It assumes the generated language uses spaces to separate words.
"""

import argparse
import os

import ctranslate2
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Model directory")
    args = parser.parse_args()
    model_dir = args.model_dir

    generator = ctranslate2.Generator(model_dir, device="cuda")

    # or load on CPU:
    # generator = ctranslate2.Generator(model_dir, device="cpu", intra_threads=6)

    sp = spm.SentencePieceProcessor(os.path.join(model_dir, "ice_text.model"))

    prompt = "What is the meaning of life?"

    print_inline(prompt)

    for word in generate_words(generator, sp, prompt):
        print_inline(word)

    print("")


def generate_words(generator, sp, prompt, add_bos=True):
    prompt_tokens = sp.encode(prompt, out_type=str)

    if add_bos:
        prompt_tokens.insert(0, "<s>")

    step_results = generator.generate_tokens(
        prompt_tokens,
        sampling_temperature=0.8,
        sampling_topk=20,
        max_length=512,
    )

    output_ids = []

    for step_result in step_results:
        is_new_word = step_result.token.startswith("▁")

        if is_new_word and output_ids:
            yield " " + sp.decode(output_ids)
            output_ids = []

        output_ids.append(step_result.token_id)

    if output_ids:
        yield " " + sp.decode(output_ids)


def print_inline(text):
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
