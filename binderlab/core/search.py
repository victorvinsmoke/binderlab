"""Motor de filtrado y búsqueda sobre las tablas de métricas.

Reglas clásicas (columna/operador/valor), expresiones libres estilo pandas,
columnas calculadas, ordenación e inyección de anotaciones. Módulo puro: solo pandas.
"""
from itertools import zip_longest
import pandas as pd


def coerce_numeric_columns(df):
    """Devuelve una copia de `df` convirtiendo a numérico las columnas que no
    pierden información al convertirse."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            continue
        conv = pd.to_numeric(out[c], errors='coerce')
        if conv.notna().sum() >= out[c].notna().sum():
            out[c] = conv
    return out


OPERATORS = [
    {'label': '≥  (mayor o igual)', 'value': '>='},
    {'label': '≤  (menor o igual)', 'value': '<='},
    {'label': '>  (mayor)',         'value': '>'},
    {'label': '<  (menor)',         'value': '<'},
    {'label': '==  (igual)',        'value': '=='},
    {'label': '!=  (distinto)',     'value': '!='},
    {'label': 'contiene',           'value': 'contains'},
    {'label': 'no contiene',        'value': 'not_contains'},
    {'label': 'empieza por',        'value': 'startswith'},
]
NUMERIC_OPS = {'>', '<', '>=', '<=', '==', '!='}


def coerce_value(op, raw_val):
    """Convierte el valor introducido al tipo requerido por el operador."""
    if op in NUMERIC_OPS:
        try:
            return float(raw_val)
        except (TypeError, ValueError):
            return None
    return str(raw_val) if raw_val is not None else None


def apply_rule(df, col, op, val):
    """Aplica una regla a un DataFrame y devuelve la máscara booleana (o None)."""
    if col not in df.columns:
        return None
    series = df[col]
    if op in NUMERIC_OPS:
        if val is None:
            return None
        s = pd.to_numeric(series, errors='coerce')
        try:
            if op == '>=': return s >= val
            if op == '<=': return s <= val
            if op == '>':  return s >  val
            if op == '<':  return s <  val
            if op == '==': return s == val
            if op == '!=': return s != val
        except Exception:
            return None
    s_str = series.fillna('').astype(str)
    if op == 'contains':
        return s_str.str.contains(str(val), na=False, case=False, regex=False)
    if op == 'not_contains':
        return ~s_str.str.contains(str(val), na=False, case=False, regex=False)
    if op == 'startswith':
        return s_str.str.startswith(str(val), na=False)
    return None


ATRIBUTOS_PROHIBIDOS = ('__', 'globals', 'builtins', 'import')


def expresion_insegura(expr):
    """Devuelve el motivo por el que la expresión no es aceptable, o None."""
    texto = str(expr)
    for patron in ATRIBUTOS_PROHIBIDOS:
        if patron in texto:
            return f"la expresión contiene '{patron}', no permitido"
    return None


def apply_expression(df, expr):
    """Evalúa una expresión booleana estilo pandas sobre `df`.
    Devuelve `(mask, error)`."""
    if not expr or not str(expr).strip():
        return None, "expresión vacía"
    motivo = expresion_insegura(expr)
    if motivo:
        return None, motivo
    df_eval = coerce_numeric_columns(df)
    try:
        result = df_eval.eval(str(expr), engine='python')
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not hasattr(result, 'dtype') or not pd.api.types.is_bool_dtype(result):
        return None, ("la expresión no devuelve una condición booleana "
                      "(usa comparadores: >=, <=, ==, and, or)")
    if isinstance(result, pd.Series):
        mask = result.reindex(df.index).fillna(False).astype(bool)
    else:
        mask = pd.Series(result, index=df.index).fillna(False).astype(bool)
    return mask, None


def apply_computed_columns(df, computed):
    """Añade al DataFrame las columnas computadas `{'name', 'expr'}` en orden,
    permitiendo que una columna computada referencie otras previas."""
    if not computed:
        return df
    df = df.copy()
    df_num = coerce_numeric_columns(df)
    for item in computed:
        name = (item or {}).get('name')
        expr = (item or {}).get('expr')
        if not name or not expr:
            continue
        motivo = expresion_insegura(expr)
        if motivo:
            print(f"[BinderMonitor] Columna '{name}' rechazada: {motivo}")
            continue
        try:
            col_val = df_num.eval(str(expr), engine='python')
            df[name] = col_val
            df_num[name] = pd.to_numeric(col_val, errors='coerce') if not pd.api.types.is_numeric_dtype(col_val) else col_val
        except Exception as e:
            print(f"[BinderMonitor] Columna computada '{name}' falló: {e}")
    return df


def inject_annotation_columns(df, annotations):
    """Añade las columnas `_tags` y `_note` cruzando por `base_name`."""
    if not annotations or 'base_name' not in df.columns:
        return df
    df = df.copy()
    df['_tags'] = df['base_name'].map(
        lambda bn: ' '.join(annotations.get(bn, {}).get('tags', [])))
    df['_note'] = df['base_name'].map(
        lambda bn: annotations.get(bn, {}).get('note', ''))
    return df


def evaluate_filter_rules(df, cols, ops, vals, modes, exprs):
    """Aplica todas las reglas (clásicas y expresiones) y devuelve `(mask, funnel_lines)`."""
    funnel_lines = [f"[*] Iniciales: {len(df)}"]
    mask_global = pd.Series(True, index=df.index)

    for col, op, val, mode, expr in zip_longest(
            cols or [], ops or [], vals or [], modes or [], exprs or []):
        if mode == 'expr':
            if not expr or not str(expr).strip():
                continue
            m, err = apply_expression(df, expr)
            if err:
                funnel_lines.append(f"  [!] expr «{expr}» → {err}. Regla ignorada.")
                continue
            before = int(mask_global.sum())
            mask_global &= m
            survivors = int(mask_global.sum())
            funnel_lines.append(f"  → expr «{expr}»   supervivientes: {survivors}")
            if survivors == 0 and before > 0:
                funnel_lines.append("     ⚠ 0 supervivientes tras esta expresión.")
            elif before > 0 and survivors / before < 0.05:
                pct = 100 * (1 - survivors / before)
                funnel_lines.append(f"     ⚠ esta expresión eliminó el {pct:.0f}% restante.")
            continue

        if not col or not op or val is None or val == '':
            continue
        coerced = coerce_value(op, val)
        if op in NUMERIC_OPS and coerced is None:
            funnel_lines.append(
                f"  [!] [{col} {op} {val}] → valor no numérico, regla ignorada.")
            continue
        m = apply_rule(df, col, op, coerced)
        if m is None:
            funnel_lines.append(
                f"  [!] [{col} {op} {val}] → operador no aplicable, regla ignorada.")
            continue

        before = int(mask_global.sum())
        mask_global &= m.fillna(False)
        survivors = int(mask_global.sum())
        funnel_lines.append(f"  → [{col} {op} {val}]   supervivientes: {survivors}")

        col_num = pd.to_numeric(df[col], errors='coerce')
        if survivors == 0 and before > 0:
            if op in ('>=', '>') and col_num.notna().any():
                sug = round(float(col_num.quantile(0.10)), 3)
                funnel_lines.append(
                    f"     ⚠ 0 supervivientes. Máximo de '{col}' = "
                    f"{col_num.max():.3g}. Prueba con ≥ {sug} (p10).")
            elif op in ('<=', '<') and col_num.notna().any():
                sug = round(float(col_num.quantile(0.90)), 3)
                funnel_lines.append(
                    f"     ⚠ 0 supervivientes. Mínimo de '{col}' = "
                    f"{col_num.min():.3g}. Prueba con ≤ {sug} (p90).")
            else:
                funnel_lines.append(
                    "     ⚠ 0 supervivientes tras esta regla. Revisa el valor.")
        elif before > 0 and survivors / before < 0.05:
            pct = 100 * (1 - survivors / before)
            funnel_lines.append(
                f"     ⚠ esta regla eliminó el {pct:.0f}% de los diseños "
                f"que quedaban. ¿Umbral demasiado estricto?")

    return mask_global, funnel_lines


def sort_dataframe(df, sort_by, sort_asc=True):
    """Ordena `df` por `sort_by` (numéricamente cuando es posible)."""
    if not sort_by or sort_by not in df.columns or df.empty:
        return df
    out = df.copy()
    conv = pd.to_numeric(out[sort_by], errors='coerce')
    if conv.notna().sum() >= out[sort_by].notna().sum():
        out[sort_by] = conv
    return out.sort_values(by=sort_by, ascending=bool(sort_asc), na_position='last')
