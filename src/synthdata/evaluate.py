"""Fidelity and privacy metrics for a synthetic table.

Fidelity answers "does the synthetic data look like the real data" and privacy
answers "does it give away individual real records". They pull in opposite
directions, so both are reported together; a generator that wins on one and
loses on the other is not a good generator.

Fidelity
    * per-column Kolmogorov-Smirnov statistic for numeric columns,
    * absolute delta of the numeric correlation matrix,
    * total-variation distance per categorical column.

Privacy
    * distance to the closest record (DCR): for each synthetic row, the L2
      distance to its nearest real row after scaling each numeric column to unit
      range. A generator that copies rows scores ~0.
    * duplicate rate: share of synthetic rows that exactly reproduce a real row,
      plus the share of internal duplicates (a model that has collapsed).
"""
