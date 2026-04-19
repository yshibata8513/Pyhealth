pyhealth.models.BulkRNABert
===========================

BulkRNABert is a Transformer-based masked-language-model (MLM) pre-trained on
bulk RNA-seq gene-expression profiles, following Gelard et al. (2024,
`arXiv:2412.11958 <https://arxiv.org/abs/2412.11958>`_). The encoder embeds
per-gene expression values (either binned into ``n_bins`` tokens or fed as a
continuous scalar) and adds a learned per-gene positional embedding, so the
sequence length equals the number of genes (19,062 by default for the TCGA
preprocessed corpus). After MLM pre-training the encoder is used to produce
mean-pooled ``(embed_dim,)`` sample embeddings for downstream tasks such as
:class:`~pyhealth.tasks.TCGACancerClassification5Cohort`.

This PyHealth implementation is a clean-room PyTorch port of the reference
`instadeepai/multiomics-open-research
<https://github.com/instadeepai/multiomics-open-research>`_ JAX/Haiku code.

.. autoclass:: pyhealth.models.BulkRNABertConfig
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BulkRNABert
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.BulkRNABertClassifier
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.bin_expression_values

.. autofunction:: pyhealth.models.load_expression_csv
