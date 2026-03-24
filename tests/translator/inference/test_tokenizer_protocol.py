from translator.inference.tokenizer.factory import TOKENIZER_CHOICES


def test_tokenizer_choices_include_hf_only():
    assert TOKENIZER_CHOICES == ("hf",)
