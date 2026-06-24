# Retrieval Pipeline Performance Report

This report summarizes the current sparse retrieval pipeline and the effect of the metadata-filtering changes.

## Current Pipeline

The retrieval flow is:

1. Parse the user query.
2. Resolve any structured sidebar metadata filters directly in SQLite.
3. Convert the resolved doc set into a compact bitmap and hand it to the C++ BM25 scorer.
4. Apply optional embedded bitmap filters from the query syntax.
5. Run BM25 scoring on the restricted candidate set.
6. Optionally rerank the top candidates.
7. Hydrate only the final documents from SQLite.

## What Metadata Filtering Changes

Metadata filtering now acts as an early candidate-reduction step instead of a post-processing step.

- Filters are parsed into an AST.
- Structured sidebar controls are resolved directly in SQLite before scoring.
- The pipeline passes doc indices downstream, then serializes them into a bitmap for the C++ scorer.
- Repeated identical filters are cached per pipeline instance.
- The sidebar now builds structured filter payloads from controls, which reduces syntax mistakes and speeds up common filtering tasks.

## Performance Impact

The main performance benefit is lower work per query.

- Selective filters reduce the number of documents that BM25 needs to score.
- The C++ scorer now receives the metadata mask directly, so it can skip non-matching documents without bouncing back to Python.
- Reranking receives a smaller candidate pool, which matters because the reranker is the most expensive stage.
- Document hydration only happens for the final results, which keeps SQLite lookups bounded.
- `has_title` is filtered on the title column directly, which avoids relying on metadata JSON for a derived property.

## Correctness Impact

- `NOT` now obeys operator precedence correctly.
- Parenthesized expressions are preserved.
- The SQLite metadata path and the BM25 fallback path now agree on the filter result set.
- Structured sidebar controls generate valid structured filters for the common metadata cases.

## Tradeoffs

- Filtering is fastest when the filter is selective.
- Very broad filters behave close to the no-filter case, with a small extra cost for resolving the metadata set.
- Cache reuse improves repeated searches with the same filter, but it assumes the corpus is stable for the lifetime of the pipeline object.

## Practical Outcome

The pipeline is now more predictable and cheaper to run:

- less unnecessary scoring,
- less reranker load,
- less repeated metadata work,
- and cleaner separation between filtering, ranking, and hydration.
