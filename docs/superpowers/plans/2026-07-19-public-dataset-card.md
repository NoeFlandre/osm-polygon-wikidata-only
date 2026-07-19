# Public dataset card implementation plan

1. Characterize and independently audit Wikipedia coverage joins on fixtures and real
   data; fail on unresolved identifiers.
2. Reproduce the antimeridian artifact in a failing geometry test, then replace the
   open-fragment split with deterministic polygon clipping.
3. Add tested loaders and aggregation for Wikipedia-or-Wikivoyage polygon coverage,
   followed by a deterministic static map.
4. Add tested Natural Earth continent assignment and factual per-continent aggregate
   rendering, including an explicit Unassigned bucket.
5. Rewrite only the public dataset-card prose and statistics labels requested by the
   user; document the token estimate as `characters / 4`, rounded down per row with a
   minimum of one token for non-empty text.
6. Integrate the new asset and statistics into every existing metadata refresh path,
   update stable remote paths and publication contracts, and regenerate goldens.
7. Run all quality gates, generate and visually inspect real assets, publish the
   updated card atomically, and verify live Hub links.
