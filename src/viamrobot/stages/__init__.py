"""Implementations of the pipeline stages, grouped by where the work actually happens.

    scripted.py  Scripted*  no model and no hardware: answers come from a scenario,
                            or from what the real models said when an image was captured
    local.py     Local*     the downloaded model files in models/, run on this computer
    machine.py   Machine*   services on the Viam machine (armfarm01), over the network

The class name says which model runs and where, for example:

    ScriptedAnomalyScorer   runs nothing; returns a score it was given
    LocalPatchCoreScorer    runs PatchCore (models/sortit-patchcore) here, with OpenVINO
    MachinePatchCoreScorer  asks vision-1 on armfarm01, which runs PatchCore on the NUC GPU

Every stage object also has a `runs` string saying the same thing in words, and
Pipeline.describe() lists them for a whole pipeline.
"""
