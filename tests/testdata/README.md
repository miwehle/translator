# Test Data

Store mapped Arrow datasets for `train_prod` tests in this directory.

Expected structure example:

- `tests/testdata/europarl_de-en_train_1000/europarl.mapped`

The loss-progress test auto-discovers the first `*.mapped` directory below
`tests/testdata`. You can override discovery with:

- `TRANSLATOR2_TESTDATA_MAPPED=<absolute-or-relative-path-to-*.mapped-dir>`
