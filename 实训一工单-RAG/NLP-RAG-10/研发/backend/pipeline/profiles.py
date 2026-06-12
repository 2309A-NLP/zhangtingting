from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProspectusProfile:
    profile_id: str
    pdf_name: str
    doc_name: str
    company_aliases: tuple[str, ...]
    artifact_dirname: str
    parse_version: str
    collection_name: str
    text_collection_name: str
    visual_collection_name: str
    mongo_collection_name: str

    def default_artifact_dir(self, project_root: Path) -> Path:
        return project_root / "artifacts" / self.artifact_dirname


PROSPECTUS_PROFILES: dict[str, ProspectusProfile] = {
    "prospectus1": ProspectusProfile(
        profile_id="prospectus1",
        pdf_name="招股说明书1.pdf",
        doc_name="招股说明书1",
        company_aliases=(
            "武汉兴图新科电子股份有限公司",
            "武汉兴图新科",
            "兴图新科",
        ),
        artifact_dirname="stage2_precise_extraction_rewire_test",
        parse_version="parse4_p1",
        collection_name="prospectus_chunks_04",
        text_collection_name="prospectus_chunks_04_text",
        visual_collection_name="prospectus_chunks_04_visual",
        mongo_collection_name="prospectus_tables_04",
    ),
    "prospectus2": ProspectusProfile(
        profile_id="prospectus2",
        pdf_name="招股说明书2.pdf",
        doc_name="招股说明书2",
        company_aliases=(
            "武汉力源信息技术股份有限公司",
            "武汉力源信息",
            "力源信息",
        ),
        artifact_dirname="prospectus2_parse4_full",
        parse_version="parse4_p2",
        collection_name="prospectus2_chunks_04",
        text_collection_name="prospectus2_chunks_04_text",
        visual_collection_name="prospectus2_chunks_04_visual",
        mongo_collection_name="prospectus2_tables_04",
    ),
}


def get_prospectus_profile(profile_id: str) -> ProspectusProfile:
    normalized = (profile_id or "").strip().lower()
    if normalized not in PROSPECTUS_PROFILES:
        raise KeyError(f"Unknown prospectus profile: {profile_id}")
    return PROSPECTUS_PROFILES[normalized]
