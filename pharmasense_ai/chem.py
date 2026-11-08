"""Molecular structure, descriptors and structural similarity (RDKit).

Similarity is reported under several fingerprints at once, because a single Tanimoto number hides how
representation-dependent it is; and every "similar drug" carries the caveat that near-identical structures can
differ sharply in activity ("activity cliffs") - similarity proposes hypotheses, it does not prove them.
"""
from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

FP_KINDS = ("morgan2", "featmorgan2", "atompair", "torsion")


@lru_cache(maxsize=2048)
def mol(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


@lru_cache(maxsize=4)
def _gen(kind: str):
    if kind == "morgan2":
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    if kind == "featmorgan2":
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                         atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    if kind == "atompair":
        return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    if kind == "torsion":
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
    raise ValueError(kind)


def fingerprint(smiles: str, kind: str = "morgan2"):
    m = mol(smiles)
    return _gen(kind).GetFingerprint(m) if m is not None else None


def similarity_profile(smiles_a: str, smiles_b: str) -> dict:
    """Tanimoto under each fingerprint plus their mean. Empty dict if either SMILES cannot be parsed."""
    out = {}
    for k in FP_KINDS:
        fa, fb = fingerprint(smiles_a, k), fingerprint(smiles_b, k)
        if fa is None or fb is None:
            return {}
        out[k] = float(DataStructs.TanimotoSimilarity(fa, fb))
    out["mean"] = sum(out[k] for k in FP_KINDS) / len(FP_KINDS)
    return out


def rank_similar(query_smiles: str, library: dict[str, str], top_k: int = 8, min_sim: float = 0.25,
                 exclude: set[str] | None = None) -> list[dict]:
    """Most similar library molecules (by mean Tanimoto over all fingerprints). library: name -> SMILES."""
    rows = []
    for name, smi in library.items():
        if exclude and name in exclude:
            continue
        prof = similarity_profile(query_smiles, smi)
        if prof and prof["mean"] >= min_sim:
            rows.append({"name": name, "smiles": smi, **prof})
    return sorted(rows, key=lambda r: -r["mean"])[:top_k]


def descriptors(smiles: str) -> dict:
    m = mol(smiles)
    if m is None:
        return {}
    mw, logp = Descriptors.MolWt(m), Descriptors.MolLogP(m)
    hbd, hba = Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m)
    return {"mw": round(mw, 2), "logp": round(logp, 2), "tpsa": round(rdMolDescriptors.CalcTPSA(m), 1),
            "hbd": hbd, "hba": hba, "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(m),
            "rings": rdMolDescriptors.CalcNumRings(m), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
            "heavy_atoms": m.GetNumHeavyAtoms(), "formal_charge": Chem.GetFormalCharge(m),
            "lipinski_violations": int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)}


def scaffold(smiles: str) -> str:
    m = mol(smiles)
    if m is None:
        return ""
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return ""


def shared_substructure_note(smiles_a: str, smiles_b: str) -> str:
    """Do the two drugs share a Murcko scaffold? One line, used in explanations."""
    sa, sb = scaffold(smiles_a), scaffold(smiles_b)
    if sa and sa == sb:
        return f"Same core scaffold ({sa})."
    return ""


def svg(smiles: str, size=(320, 220)) -> str:
    """2-D structure as SVG, or "" when it cannot be drawn. The drawing module is imported here, not at the top of the file:
    on some servers it needs system libraries (libXrender) that are missing, and everything else in this module
    (fingerprints, similarity, descriptors, scaffolds) works without it."""
    m = mol(smiles)
    if m is None:
        return ""
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return ""
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    d.drawOptions().clearBackground = False
    d.DrawMolecule(m)
    d.FinishDrawing()
    return d.GetDrawingText()
