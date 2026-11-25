"""Motor de filtrado y busqueda sobre las tablas de metricas.

Reglas clasicas (columna/operador/valor), expresiones libres estilo pandas,
columnas calculadas e inyeccion de anotaciones. Modulo puro: solo pandas.
"""
import pandas as pd


def coerce_numeric_columns(df):
    """Devuelve una copia de `df` con las columnas convertidas a numerico
    cuando se puede sin perder informacion.

    Las tablas que vienen del cache traen numeros guardados como texto; sin
    esta conversion, comparar `pae < 10` compararia cadenas. Solo se
    sustituye la columna si la conversion no introduce NaN nuevos, para no
    destruir columnas de texto legitimas.

    (Antes este bucle estaba duplicado en apply_expression y en
    apply_computed_columns.)
    """
    out = df.copy()
    for c in out.columns:
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
    """Convierte el valor introducido en la UI al tipo correcto según el
    operador. Si el operador es numérico y el valor no se puede convertir
    a float, devuelve None (la regla se descartará)."""
    if op in NUMERIC_OPS:
        try:
            return float(raw_val)
        except (TypeError, ValueError):
            return None
    return str(raw_val) if raw_val is not None else None


def apply_rule(df, col, op, val):
    """Aplica una regla a un DataFrame y devuelve la máscara booleana.
    Devuelve None si la columna no existe o el operador no es aplicable."""
    if col not in df.columns:
        return None
    series = df[col]
    if op in NUMERIC_OPS:
        # Forzar la columna a numérica; valores no convertibles → NaN
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
    s_str = series.astype(str)
    if op == 'contains':
        return s_str.str.contains(str(val), na=False, case=False)
    if op == 'not_contains':
        return ~s_str.str.contains(str(val), na=False, case=False)
    if op == 'startswith':
        return s_str.str.startswith(str(val), na=False)
    return None


# Atributos que nunca deben aparecer en una expresion de usuario. pandas.eval
# bloquea las llamadas a funciones no permitidas, asi que NO hay ejecucion
# arbitraria; pero el acceso a atributos dunder si funciona y filtra objetos
# internos (p.ej. `col.__class__.__init__.__globals__` devuelve un dict).
# Como ninguna expresion legitima de filtrado los necesita, se rechazan.
ATRIBUTOS_PROHIBIDOS = ('__', 'globals', 'builtins', 'import')


def expresion_insegura(expr):
    """Devuelve el motivo por el que la expresion no es aceptable, o None."""
    texto = str(expr)
    for patron in ATRIBUTOS_PROHIBIDOS:
        if patron in texto:
            return f"la expresión contiene '{patron}', no permitido"
    return None


def apply_expression(df, expr):
    """Evalúa una expresión de texto libre estilo pandas sobre el DataFrame
    (m01 — modo expresión). Equivalente al modo `df.eval()` que ya soporta
    extract_hits.py por CLI.

    Devuelve (mask, error). Si la expresión compila y produce una máscara
    booleana, error=None. Si falla (columna inexistente, sintaxis inválida,
    o resultado no booleano), mask=None y error=mensaje legible.

    Para que las columnas numéricas guardadas como texto en el cache se
    comparen bien, se intenta convertir a numérico todo lo que se pueda
    antes de evaluar (sin destruir las columnas de texto)."""
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
    if not hasattr(result, 'dtype') or result.dtype != bool:
        return None, ("la expresión no devuelve una condición booleana "
                      "(usa comparadores: >=, <=, ==, and, or)")
    return result.reindex(df.index).fillna(False), None


def apply_computed_columns(df, computed):
    """Añade al DataFrame las columnas computadas definidas por el usuario
    (m09). `computed` es una lista de dicts {'name', 'expr'}. Cada expresión
    se evalúa con df.eval() sobre las columnas numéricas; los fallos se
    ignoran silenciosamente (la columna simplemente no se añade).

    Devuelve el DataFrame (posiblemente con columnas nuevas). No muta el
    original cuando hay columnas que añadir (trabaja sobre una copia)."""
    if not computed:
        return df
    df = df.copy()
    # Versión numérica para evaluar de forma fiable
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
            df[name] = df_num.eval(str(expr), engine='python')
        except Exception as e:
            print(f"[BinderMonitor] Columna computada '{name}' falló: {e}")
    return df


def inject_annotation_columns(df, annotations):
    """Añade las columnas `_tags` (texto con los tags separados por ' ') y
    `_note` (texto libre) al DataFrame, cruzando por `base_name`.

    `annotations` es el dict {design_name: {'tags', 'note'}}. Si el DataFrame
    no tiene `base_name` o no hay anotaciones, se devuelve sin cambios.
    Tener estas columnas hace que las anotaciones sean filtrables como
    cualquier otra: «_tags contains ⭐» es una regla normal."""
    if not annotations or 'base_name' not in df.columns:
        return df
    df = df.copy()
    df['_tags'] = df['base_name'].map(
        lambda bn: ' '.join(annotations.get(bn, {}).get('tags', [])))
    df['_note'] = df['base_name'].map(
        lambda bn: annotations.get(bn, {}).get('note', ''))
    return df
