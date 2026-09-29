Feature: Offline polygon extraction from a PBF
  Executable specification for issue #115: extraction runs against a tiny,
  locally written PBF and in-memory enrichment clients, never the network.

  Scenario: A way and a multi-QID multipolygon relation become polygon rows
    Given a tiny PBF with a tagged way and a multipolygon relation tagged "Q1;Q2"
    When I run extraction with in-memory enrichment clients
    Then the polygon Parquet has one row per tagged polygon
    And every polygon row has a valid polygon_id and a positive area
    And the relation row keeps both QIDs "Q1" and "Q2"

  Scenario: An untagged polygon is excluded
    Given a tiny PBF with a tagged way and a multipolygon relation tagged "Q1;Q2"
    When I run extraction with in-memory enrichment clients
    Then the untagged closed way is not in the polygon Parquet
