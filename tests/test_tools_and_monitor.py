"""Tests de herramientas CLI (dna, corr, score) y lógica del dashboard monitor."""
import base64
import pandas as pd
import pytest

from binderlab import aa_to_dna, correlaciones, scoring_fisico
import binderlab.monitor as monitor


def test_dna_validaciones_sin_deps():
    """optimize_for_idt rechaza secuencias vacías o con caracteres inválidos antes de cargar dnachisel."""
    r_empty = aa_to_dna.optimize_for_idt("   *  ")
    assert r_empty["dna"] is None
    assert "vacía" in r_empty["error"]

    r_bad = aa_to_dna.optimize_for_idt("MAAA_BX_KKK")
    assert r_bad["dna"] is None
    assert "B" in r_bad["error"] and "X" in r_bad["error"]


def test_dna_read_input_bom_tsv_fasta(tmp_path):
    """_read_input soporta CSV con BOM UTF-8, TSV, FASTA y avisa si falta seq_col."""
    csv_bom = tmp_path / "hits.csv"
    csv_bom.write_bytes("\ufeffdescription,sequence\ndes_1,MAAAKKK\n".encode("utf-8"))
    rows = aa_to_dna._read_input(csv_bom, "sequence", "description")
    assert rows == [("des_1", "MAAAKKK")]

    tsv = tmp_path / "hits.tsv"
    tsv.write_text("description\tsequence\ndes_2\tMEEEKKK\n", encoding="utf-8")
    assert aa_to_dna._read_input(tsv, "sequence", "description") == [("des_2", "MEEEKKK")]

    fa = tmp_path / "hits.fasta"
    fa.write_text(">des_3 info extra\nmaaa\nkkke\n", encoding="utf-8")
    assert aa_to_dna._read_input(fa, "sequence", "description") == [("des_3", "MAAAKKKE")]

    with pytest.raises(ValueError, match="no encontrada"):
        aa_to_dna._read_input(csv_bom, "col_inexistente", "description")


def test_correlaciones_heuristicas_sin_fragmentar():
    """calcular_descriptores_heuristas calcula span, entropía y rareza correctamente."""
    df = pd.DataFrame({
        "hotspots": ["A10,A25", "A10", ""],
        "bin_hotspots_A10": [1, 1, 0],
        "bin_hotspots_A25": [1, 0, 0],
    })
    out, cols = correlaciones.calcular_descriptores_heuristas(
        df, "hotspots", ["bin_hotspots_A10", "bin_hotspots_A25"]
    )
    assert cols == ["hotspots_span", "hotspots_entropy", "hotspots_rareza"]
    assert list(out["hotspots_span"]) == [15, 0, 0]
    assert out["hotspots_entropy"].iloc[1] == 1.0


def test_scoring_merge_af2_y_residuos_terminales(tmp_path):
    """_merge_af2_scores empareja con o sin _af2pred y limpia cabeceras repetidas."""
    rosetta_df = pd.DataFrame({
        "description": ["des_0", "des_1_af2pred"],
        "ddG": [-32.1, -28.4],
    })
    sc_file = tmp_path / "out.sc"
    sc_file.write_text(
        "SCORE: pae_interaction plddt_total description\n"
        "SCORE: 5.0 90.0 des_0_af2pred\n"
        "SCORE: pae_interaction plddt_total description\n"
        "SCORE: 8.0 0.0 des_1\n",
        encoding="utf-8",
    )
    merged = scoring_fisico._merge_af2_scores(rosetta_df, sc_file)
    assert len(merged) == 2
    assert round(float(merged.loc[0, "pae_plddt_ratio"]), 4) == round(5.0 / 0.9, 4)
    assert pd.isna(merged.loc[1, "pae_plddt_ratio"])

    class _DummyRes:
        def __init__(self, apolar, name, name3):
            self._apolar = apolar
            self._name = name
            self._name3 = name3

        def is_apolar(self):
            return self._apolar

        def name(self):
            return self._name

        def name3(self):
            return self._name3

    # Un residuo aromático N-terminal con patch debe reconocerse como hidrofóbico
    res_nterm_tyr = _DummyRes(False, "TYR:NtermProteinFull", "TYR")
    assert scoring_fisico._is_hydrophobic_residue(res_nterm_tyr) is True


def test_monitor_manifest_roundtrip_y_radar():
    """export_manifest -> load_manifest restaura fielmente reglas clásicas y expresiones."""
    manifest_out = monitor.export_manifest(
        1,
        cols=["pae_interaction", None],
        ops=["<=", ">="],
        vals=["10", ""],
        modes=["classic", "expr"],
        exprs=["", "plddt_binder >= 85"],
        computed=[{"name": "eff", "expr": "ddG / 100"}],
        sort_by="eff",
        sort_asc=True,
        metric_file="/tmp/out.sc",
        annotations={},
    )
    b64 = base64.b64encode(manifest_out["content"].encode("utf-8")).decode("ascii")
    rules, next_id, computed, sort_by, sort_asc, msg, cleared = monitor.load_manifest(
        f"data:application/json;base64,{b64}", "manifest.json", counter=5
    )
    assert cleared is None
    assert next_id == 7
    assert len(rules) == 2
    assert rules[0] == {
        "id": 5, "mode": "classic", "col": "pae_interaction",
        "op": "<=", "val": "10", "expr": "",
    }
    assert rules[1] == {
        "id": 6, "mode": "expr", "col": None,
        "op": ">=", "val": "", "expr": "plddt_binder >= 85",
    }
    assert computed == [{"name": "eff", "expr": "ddG / 100"}]
    assert sort_by == "eff" and sort_asc is True
    assert "2 regla(s)" in msg

    # Renderizar las reglas restauradas no falla y respeta ambos modos
    rendered = monitor.render_rules(rules, ["pae_interaction", "plddt_binder"], computed, {})
    assert len(rendered) == 2

    # Radar ignora NaNs y maneja empates al 50%
    cache = {
        "records": [
            {"base_name": "h1", "plddt_binder": 80.0, "pae_interaction": 10.0},
            {"base_name": "h2", "plddt_binder": 80.0, "pae_interaction": 20.0},
            {"base_name": "h3", "plddt_binder": None, "pae_interaction": None},
        ]
    }
    fig, _ = monitor.update_radar("h1", cache)
    # plddt_binder empata (80 vs 80) -> radio 0.5; pae_interaction es el mejor de 2 (10 < 20) -> 0.75
    r_vals = list(fig.data[0].r)
    theta_vals = list(fig.data[0].theta)
    by_metric = dict(zip(theta_vals[:-1], r_vals[:-1]))
    assert by_metric["plddt_binder"] == 0.5
    assert by_metric["pae_interaction"] == 0.75


def test_monitor_exp_label_y_custom_stages(tmp_path):
    """_exp_label evita '.' y parse_pipeline_state soporta etapas personalizadas con RF."""
    exp = tmp_path / "campana_A"
    exp.mkdir()
    assert monitor._exp_label(str(exp), str(exp)) == "campana_A"

    old_pl = monitor.PIPELINE
    try:
        monitor.PIPELINE = monitor.load_pipeline_structure(None)
        monitor.PIPELINE["stages"] = [
            {"key": "rfdiffusion", "label": "RF", "dir": "RF_output"},
            {"key": "custom_fold", "label": "CustomFold", "dir": "CF_output"},
        ]
        monitor._init_stages()
        log_file = exp / "run.log"
        log_file.write_text(
            "[1/3] Running AlphaFold2\n"
            "finished AF2 prediction in 3.2 seconds\n",
            encoding="utf-8",
        )
        state = monitor.parse_pipeline_state(str(log_file), str(exp))
        assert set(state["stages"]) == {"rfdiffusion", "custom_fold"}
    finally:
        monitor.PIPELINE = old_pl
        if old_pl is not None:
            monitor._init_stages()
