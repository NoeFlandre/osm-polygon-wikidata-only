Feature: Resumable V2 sentence splitting

  Scenario: Resume after a segmentation failure without losing language or offsets
    Given a local V2 region with supported and unsupported language sections
    When sentence splitting fails during the second supported-language batch
    Then the interrupted run leaves the source unchanged, a durable checkpoint, and no final output
    When I resume sentence splitting
    Then resumed rows preserve exact text, offsets, and language routing
    And the manifest records supported and unsupported languages
    When I run the same input cleanly
    Then the resumed and clean Parquet outputs are identical
