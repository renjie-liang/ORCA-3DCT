"""Clinical ontology loader for mask-derived DTBD3D importance maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


CLINICAL_FINDINGS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]

SPATIAL_EXTENTS = {"focal", "moderate", "diffuse"}


@dataclass(frozen=True)
class Finding:
    name: str
    ts_class_ids: frozenset[int]
    spatial_extent: str


@dataclass(frozen=True)
class ClinicalOntology:
    ts_class_id: dict[str, int]
    organ_alias: dict[str, list[str]]
    findings: list[Finding]

    def finding_to_ts_ids(self, finding_name: str) -> set[int]:
        for finding in self.findings:
            if finding.name == finding_name:
                return set(finding.ts_class_ids)
        raise KeyError(finding_name)

    def union_relevant_ts_ids(self) -> set[int]:
        out: set[int] = set()
        for finding in self.findings:
            out |= set(finding.ts_class_ids)
        return out


def _resolve_organ_ids(
    organ_names: list[str],
    organ_alias: dict[str, list[str]],
    ts_class_id: dict[str, int],
) -> frozenset[int]:
    resolved: set[int] = set()

    def resolve_one(name: str, stack: tuple[str, ...]) -> None:
        if name in ts_class_id:
            resolved.add(int(ts_class_id[name]))
            return
        if name not in organ_alias:
            raise KeyError(f"Unknown TS organ name in ontology: {name}")
        if name in stack:
            chain = " -> ".join(stack + (name,))
            raise ValueError(f"Cyclic organ alias in ontology: {chain}")
        for child in organ_alias[name]:
            resolve_one(child, stack + (name,))

    for organ_name in organ_names:
        resolve_one(organ_name, ())
    return frozenset(resolved)


def load_ontology(yaml_path: str | Path) -> ClinicalOntology:
    path = Path(yaml_path)
    raw = yaml.safe_load(path.read_text())
    ts_class_id = {str(name): int(value) for name, value in raw["ts_class_id"].items()}
    organ_alias = {
        str(name): [str(item) for item in values]
        for name, values in raw["organ_alias"].items()
    }
    findings_raw = raw["findings"]
    names = [str(item["name"]) for item in findings_raw]
    if names != CLINICAL_FINDINGS:
        raise ValueError(f"Clinical finding order mismatch: {names}")
    for name, value in ts_class_id.items():
        if value < 0 or value > 117:
            raise ValueError(f"TS class id out of range for {name}: {value}")

    findings: list[Finding] = []
    for item in findings_raw:
        extent = str(item["spatial_extent"])
        if extent not in SPATIAL_EXTENTS:
            raise ValueError(f"Invalid spatial_extent for {item['name']}: {extent}")
        organ_names = [str(name) for name in item["ts_organs"]]
        findings.append(
            Finding(
                name=str(item["name"]),
                ts_class_ids=_resolve_organ_ids(organ_names, organ_alias, ts_class_id),
                spatial_extent=extent,
            )
        )
    return ClinicalOntology(ts_class_id=ts_class_id, organ_alias=organ_alias, findings=findings)
