# Metadata Filtering Impact Report

## What Changed

Metadata filtering now happens before sparse retrieval is scored:

1. The UI builds a structured filter payload from the sidebar controls.
2. The payload is resolved directly in SQLite using document columns and JSON metadata.
3. Matching documents are reduced to a roaring-style bitset of doc indices.
4. The bitset is passed into the C++ BM25 scorer.
5. BM25 scoring runs only over that restricted candidate set.

This keeps the filtering step separate from ranking, which makes the behavior easier to reason about and faster to execute.

## Pipeline Effect

### Before

- Queries always searched the full BM25 corpus.
- Filtering was either missing, incomplete, or handled too late.
- Repeated searches with the same filter re-parsed and re-queried SQLite every time.

### After

- Metadata filters are resolved once and cached per pipeline instance.
- The retrieval layer receives doc indices directly instead of converting doc IDs repeatedly.
- BM25 scoring skips documents that do not satisfy the metadata constraints, including inside the C++ engine.
- Reranking also sees only the filtered candidate pool, so the expensive cross-encoder stage runs on fewer documents.
- Common controls such as year, category, title presence, and code presence now bypass expression parsing entirely.

## Performance Impact

The biggest gain is from shrinking the candidate set before BM25 scoring and reranking.

- Selective filters, such as `year == 2020` or `category == 'tech'`, reduce scoring work substantially.
- Less selective filters still benefit from cached resolution and cheaper mask handling.
- The improvement is most visible on larger corpora, where the cost of scoring and hydration dominates.
- Structured sidebar controls reduce invalid filter syntax and make the common cases faster to enter.

## Correctness Improvements

- `NOT` now respects operator precedence correctly.
- Parenthesized expressions are preserved.
- Missing or empty filters still behave as "no filter".
- Document index matching is now available alongside document ID matching for faster downstream use.
- The Streamlit UI now emits structured filter payloads instead of forcing raw expression entry.
- `has_title` is resolved from the document title column directly, not from JSON metadata.

## Tradeoffs

- A highly selective filter can return zero results sooner, which is good for speed but can feel stricter to users.
- Caching makes repeated identical filters faster, but stale caches would need to be cleared if the underlying corpus changes during the same process.

## Recommended Usage

- Use metadata filters for dataset constraints, year ranges, categories, or other structured fields.
- Keep query text focused on the semantic search terms.
- Prefer selective filters when the user already knows the sub-population they want.
- Use the custom rule builder only for fields not covered by the quick controls.

## Notes

The current implementation keeps the public filter syntax intact while making the backend cheaper to run. The retrieval quality for matching documents is unchanged; only the candidate pool is reduced before scoring.
