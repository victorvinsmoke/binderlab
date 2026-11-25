"""Tests unitarios de los módulos de core (lógica pura)."""
import pathlib
import pandas as pd

from binderlab.core.config import Settings, load_pipeline_structure
from binderlab.core import scanner as S
from binderlab.core import search as Q
from binderlab.core import annotations as A
from binderlab.core import pdb as P
from binderlab.core import hpc as H


def test_core_es_puro():
    """Los módulos de core/ no deben importar Dash."""
    raiz = pathlib.Path(__file__).parent.parent / "binderlab" / "core"
    for f in raiz.glob("*.py"):
        texto = f.read_text(encoding="utf-8")
        assert "import dash" not in texto, f"{f.name} importa dash"


def test_settings_y_stages():
    """Settings deriva las constantes de etapa desde el preset."""
    pl_complexa = load_pipeline_structure("complexa")
    s_complexa = Settings(master_dir="/tmp", pipeline=pl_complexa)
    assert s_complexa.stage_keys == ("complexa",)

    pl_default = load_pipeline_structure(None)
    s_default = Settings(master_dir="/tmp", pipeline=pl_default)
    assert len(s_default.stage_keys) == 3


def test_trajectory_pdb_import_re(tmp_path):
    """Localizar trayectoria multimodelo usa `re` correctamente."""
    rf_dir = tmp_path / "RF_output" / "traj"
    rf_dir.mkdir(parents=True)
    traj_file = rf_dir / "design_1_pX0_traj.pdb"
    traj_file.write_text("MODEL 1\nATOM 1\nENDMDL\n")

    pl = {
        "rfdiffusion_dir": "RF_output",
        "trajectory_subdir": "traj",
        "trajectory_suffix": "_pX0_traj"
    }
    found = P.get_trajectory_pdb(str(tmp_path), "design_1_dldesign_2", pl)
    assert found == str(traj_file)


def test_count_pdb_models(tmp_path):
    """Cuenta bloques MODEL y ENDMDL en un archivo PDB multimodelo."""
    pdb_file = tmp_path / "multi.pdb"
    pdb_file.write_text("MODEL 1\nATOM\nENDMDL\nMODEL 2\nATOM\nENDMDL\n")
    assert P.count_pdb_models(str(pdb_file)) == 2

    # Solo ENDMDL
    pdb_no_model = tmp_path / "no_model.pdb"
    pdb_no_model.write_text("ATOM\nENDMDL\nATOM\nENDMDL\nATOM\nENDMDL\n")
    assert P.count_pdb_models(str(pdb_no_model)) == 3


def test_scanner_soporta_csv_y_tsv(tmp_path):
    """scanner.py lee archivos .csv y .tsv correctamente."""
    pl = load_pipeline_structure(None)

    csv_path = tmp_path / "binder_results_0.csv"
    csv_path.write_text("id_gen,pdb_path,metric\n0,/w/exp/job_1/job_1.pdb,1.23\n")

    tsv_path = tmp_path / "scored.tsv"
    tsv_path.write_text("description\tddG\tpae_plddt_ratio\njob_1\t-25.4\t0.08\n")

    files = S.get_metric_files(str(tmp_path), pl)
    assert str(csv_path) in files
    assert str(tsv_path) in files

    df_csv = S.parse_metrics_file(str(csv_path), pl)
    assert "base_name" in df_csv.columns
    assert df_csv["base_name"].iloc[0] == "job_1"

    df_tsv = S.parse_metrics_file(str(tsv_path), pl)
    assert "base_name" in df_tsv.columns
    assert df_tsv["base_name"].iloc[0] == "job_1"
    assert df_tsv["ddG"].iloc[0] == -25.4


def test_scanner_soporta_sc(tmp_path):
    """scanner.py lee archivos .sc con whitespace variable y descarta la columna SCORE."""
    pl = load_pipeline_structure(None)
    sc_path = tmp_path / "out.sc"
    sc_path.write_text("SCORE: description pae_interaction plddt_total\nSCORE: des_0 4.5 88.2\n")
    df = S.parse_metrics_file(str(sc_path), pl)
    assert "SCORE:" not in df.columns
    assert "base_name" in df.columns
    assert df["base_name"].iloc[0] == "des_0"
    assert float(df["pae_interaction"].iloc[0]) == 4.5


def test_search_reglas_y_expresiones():
    """Motor de búsqueda evalúa operadores numéricos y filtra atributos dunder."""
    df = pd.DataFrame({"pae": [5, 12], "plddt": [90, 70], "tag": ["cand", "desc"]})

    # Regla clásica
    mask_num = Q.apply_rule(df, "pae", "<", 10.0)
    assert list(mask_num) == [True, False]

    # Expresión segura
    mask_expr, err = Q.apply_expression(df, "pae < 10 and plddt > 80")
    assert err is None
    assert list(mask_expr) == [True, False]

    # Expresión insegura bloqueada
    _, err_sec = Q.apply_expression(df, "pae.__class__.__init__.__globals__")
    assert err_sec is not None

    # Columnas computadas
    df_comp = Q.apply_computed_columns(df, [{"name": "ratio", "expr": "pae / plddt"}])
    assert "ratio" in df_comp.columns
    assert round(df_comp["ratio"].iloc[0], 3) == round(5 / 90, 3)


def test_anotaciones_sqlite(tmp_path):
    """Ciclo CRUD de anotaciones (tags y notas) en SQLite."""
    exp_dir = str(tmp_path)
    A.set_design_tags(exp_dir, "hit_1", ["⭐ candidato", "🧪 a clonar"])
    A.set_design_note(exp_dir, "hit_1", "Excelente afinidad in silico")

    tags, note = A.get_design_annotations(exp_dir, "hit_1")
    assert "⭐ candidato" in tags
    assert "🧪 a clonar" in tags
    assert note == "Excelente afinidad in silico"

    all_ann = A.get_all_annotations(exp_dir)
    assert "hit_1" in all_ann

    df = pd.DataFrame({"base_name": ["hit_1", "hit_2"]})
    df_inj = Q.inject_annotation_columns(df, all_ann)
    assert "_tags" in df_inj.columns
    assert "⭐ candidato" in df_inj["_tags"].iloc[0]
    assert df_inj["_note"].iloc[0] == "Excelente afinidad in silico"
    assert df_inj["_tags"].iloc[1] == ""


def test_hpc_status():
    """get_hpc_status devuelve (fuente, contenido) sin fallar."""
    src, content = H.get_hpc_status()
    assert src in ("SLURM", "GPU", "N/A")
    assert isinstance(content, str)


def test_resolve_structure_pdb(tmp_path):
    """resolve_structure_pdb localiza estructuras para Complexa y AF2."""
    pl = {
        "alphafold_dir": "AF_output",
        "af_suffix": "_af2pred",
        "passed_subdir": "passed",
    }

    # 1. Complexa
    job_dir = tmp_path / "job_001"
    job_dir.mkdir()
    complexa_file = job_dir / "job_001_updated.pdb"
    complexa_file.write_text("ATOM 1\nEND\n")
    assert P.resolve_structure_pdb(str(tmp_path), "job_001", pl) == str(complexa_file)

    # 2. AF2 en passed/
    af_passed = tmp_path / "AF_output" / "passed"
    af_passed.mkdir(parents=True)
    af_file = af_passed / "des_1_af2pred.pdb"
    af_file.write_text("ATOM 1\nEND\n")
    assert P.resolve_structure_pdb(str(tmp_path), "des_1", pl) == str(af_file)

    # 3. AF2 en raíz de AF_output/
    af_root_file = tmp_path / "AF_output" / "des_2_af2pred.pdb"
    af_root_file.write_text("ATOM 1\nEND\n")
    assert P.resolve_structure_pdb(str(tmp_path), "des_2", pl) == str(af_root_file)

    # 4. Inexistente
    assert P.resolve_structure_pdb(str(tmp_path), "no_existe", pl) is None

    # 5. No colisionar prefijos numéricos (design_1 vs design_10)
    (af_passed / "des_10_af2pred.pdb").write_text("ATOM 10\nEND\n")
    assert P.resolve_structure_pdb(str(tmp_path), "des_1_dldesign_1", pl) is None
    (af_passed / "des_1_dldesign_10_af2pred.pdb").write_text("ATOM 10\nEND\n")
    assert P.resolve_structure_pdb(str(tmp_path), "des_1_dldesign_1", pl) is None
    (af_passed / "des_3_2_af2pred.pdb").write_text("ATOM 2\nEND\n")
    assert P.resolve_structure_pdb(str(tmp_path), "des_3", pl) is None


def test_trajectory_no_prefix_collision(tmp_path):
    """get_trajectory_pdb no confunde design_1 con design_10."""
    rf_dir = tmp_path / "RF_output" / "traj"
    rf_dir.mkdir(parents=True)
    (rf_dir / "design_10_pX0_traj.pdb").write_text("MODEL 1\nATOM 1\nENDMDL\n")
    pl = {
        "rfdiffusion_dir": "RF_output",
        "trajectory_subdir": "traj",
        "trajectory_suffix": "_pX0_traj",
    }
    assert P.get_trajectory_pdb(str(tmp_path), "design_1_dldesign_0", pl) is None


def test_sc_concatenated_headers_and_numeric(tmp_path):
    """Archivos .sc concatenados con múltiples cabeceras se limpian sin romper tipos."""
    sc_path = tmp_path / "concat.sc"
    sc_path.write_text(
        "SCORE: pae_interaction plddt_total description\n"
        "SCORE: 4.5 88.2 des_0_af2pred\n"
        "SCORE: pae_interaction plddt_total description\n"
        "SCORE: 6.1 91.0 des_1_af2pred\n"
    )
    df = S.parse_metrics_file(str(sc_path), None)
    assert len(df) == 2
    assert list(df["base_name"]) == ["des_0", "des_1"]
    assert pd.api.types.is_numeric_dtype(df["pae_interaction"])

    # Una columna numérica legítima llamada SCORE en un CSV no se descarta
    csv_score = tmp_path / "docking.csv"
    csv_score.write_text("description,SCORE\ndes_0,-12.5\n")
    df_csv = S.parse_metrics_file(str(csv_score), None)
    assert "SCORE" in df_csv.columns
    assert float(df_csv["SCORE"].iloc[0]) == -12.5


def test_search_literal_contains_and_chained_computed():
    """contains trata caracteres regex como literales y computed soporta encadenamiento."""
    df = pd.DataFrame({
        "a": ["10", "20"],
        "b": ["2", "5"],
        "_tags": ["🔬 in vitro+", "🔬 in vitro (pendiente"],
    })
    m_plus = Q.apply_rule(df, "_tags", "contains", "🔬 in vitro+")
    assert list(m_plus) == [True, False]

    # Un paréntesis sin cerrar no debe lanzar re.error
    m_paren = Q.apply_rule(df, "_tags", "contains", "(pendiente")
    assert list(m_paren) == [False, True]

    # Columnas computadas encadenadas: c2 depende de c1
    df_c = Q.apply_computed_columns(df, [
        {"name": "c1", "expr": "a / b"},
        {"name": "c2", "expr": "c1 * 10"},
    ])
    assert list(df_c["c1"]) == [5.0, 4.0]
    assert list(df_c["c2"]) == [50.0, 40.0]

    # evaluate_filter_rules ignora valores no numéricos en != igual que en UI
    mask, _ = Q.evaluate_filter_rules(
        df_c, ["c1"], ["!="], ["no_num"], ["classic"], [""]
    )
    assert list(mask) == [True, True]

    # sort_dataframe ordena numéricamente aunque la columna venga como texto
    df_sort = Q.sort_dataframe(pd.DataFrame({"x": ["100", "20", "3"]}), "x", True)
    assert list(df_sort["x"]) == [3, 20, 100]
