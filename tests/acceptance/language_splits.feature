Feature: V2 row-level language splits
  Executable specification for issue #115.

  Scenario: Each language shard holds only its own rows and reruns are byte-identical
    Given a V2 region with French, German and unusable-language rows
    When I build the language splits
    Then each language shard contains only rows of its own language
    And the union of the shards equals the input rows
    When I build the language splits again
    Then the language split outputs are byte-identical

  # Regression for #26.
  Scenario: Stale shards from a previous run are removed
    Given a V2 region with French, German and unusable-language rows
    And the language splits were built once
    When the German document disappears and I rebuild the language splits
    Then the stale German document shard is removed
    And the manifest no longer references the German document shard

  # Regression for #27.
  Scenario: A split output that overlaps its source is rejected
    Given a V2 region with French, German and unusable-language rows
    When I build the language splits into the processed root itself
    Then the build is rejected for overlapping its source
    And the source Parquet files are unchanged
