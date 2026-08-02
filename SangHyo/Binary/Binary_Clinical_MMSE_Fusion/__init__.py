"""CN vs MCI+DEM binary classification from wearable summaries + MMSE.

Unlike the wearable-only experiment, this model deliberately opens the MMSE
cognitive-test source and uses the item/total scores as features.  It never
uses the diagnosis columns (DIAG_NM / DIAG_SEQ) or administrative metadata; the
label still comes only from the Gait/Sleep label copies.  See README_KO.md.
"""
