"""Pipeline de matrices de correlación y reducción dimensional sobre métricas de diseño.

Permite derivar descriptores numéricos a partir de columnas discretas o compuestas (ej. hotspots)
usando heurísticas espaciales/entropía, PCA lineal o autoencoders no lineales, correlacionándolos
con métricas físico-químicas y de plegamiento.

Uso:
  binderlab corr -i datos.csv -c hotspots -t heurísticas pca
  binderlab corr -i metricas.sc -c hotspots -t heurísticas pca
  binderlab corr -i datos.csv -c hotspots -t autoencoder
"""

import argparse
import os
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sns = None
plt = None


def _load_plotting():
    global sns, plt
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as _plt
        import seaborn as _sns
        sns, plt = _sns, _plt
    except ImportError as e:
        sys.exit(f"[!] Falta dependencia de graficado ({e}). Instala: pip install '.[corr]'")


def calcular_descriptores_heuristas(df, col_target, columnas_binarias):
    """Calcula span, entropía de Shannon y rareza sobre las posiciones activas."""
    print(f"[*] Descriptores heurísticos sobre '{col_target}'...")
    df_resultado = df.copy()
    df_bin = df_resultado[columnas_binarias]

    def extraer_numeros(texto_str):
        if pd.isna(texto_str) or texto_str in ('', 'nan'):
            return []
        return [int(n) for n in re.findall(r'\d+', str(texto_str))]

    df_resultado[f'{col_target}_span'] = df_resultado[col_target].apply(
        lambda x: max(extraer_numeros(x)) - min(extraer_numeros(x)) if len(extraer_numeros(x)) >= 2 else 0
    )

    total_posiciones = df_bin.shape[1]
    unos = df_bin.sum(axis=1)
    p1 = unos / total_posiciones
    p0 = (total_posiciones - unos) / total_posiciones
    log_p1 = np.log2(p1.replace(0, 1))
    log_p0 = np.log2(p0.replace(0, 1))
    df_resultado[f'{col_target}_entropy'] = -(p1 * log_p1 + p0 * log_p0)

    frecuencias = df_bin.mean().replace(0, 1e-5)
    pesos_rareza = -np.log(frecuencias)
    suma_rareza = df_bin.dot(pesos_rareza)
    conteo_activos = unos.replace(0, 1)
    df_resultado[f'{col_target}_rareza'] = suma_rareza / conteo_activos

    nuevos_descriptores = [f'{col_target}_span', f'{col_target}_entropy', f'{col_target}_rareza']
    return df_resultado, nuevos_descriptores


def aplicar_reduccion_pca(df, col_target, columnas_binarias):
    """Proyección PCA 1D sobre la representación binaria."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        sys.exit("[!] Falta 'scikit-learn' para PCA. Instala: pip install '.[corr]'")

    print(f"[*] Reducción PCA 1D sobre '{col_target}'...")
    df_resultado = df.copy()
    X = df_resultado[columnas_binarias].values

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    col_pca = f'{col_target}_PCA_1D'
    df_resultado[col_pca] = X_pca[:, 0]

    var_explicada = pca.explained_variance_ratio_[0] * 100
    print(f"  [+] PCA completado. Varianza explicada: {var_explicada:.1f}%")
    return df_resultado, [col_pca]


def aplicar_reduccion_autoencoder(df, col_target, columnas_binarias):
    """Compresión no lineal 1D mediante cuello de botella en autoencoder neuronal."""
    try:
        from tensorflow.keras import layers, Model
    except ImportError:
        sys.exit("[!] Falta 'tensorflow' para autoencoder. Instala: pip install '.[corr-ae]'")

    print(f"[*] Autoencoder 1D sobre '{col_target}'...")
    df_resultado = df.copy()
    X = df_resultado[columnas_binarias].values
    input_dim = X.shape[1]

    def _leaky_relu(slope=0.1):
        try:
            return layers.LeakyReLU(negative_slope=slope)
        except TypeError:
            return layers.LeakyReLU(alpha=slope)

    input_layer = layers.Input(shape=(input_dim,), name=f'Entrada_{col_target}')
    e = layers.Dense(48, name='Enc_48')(input_layer)
    e = _leaky_relu(0.1)(e)
    e = layers.Dropout(0.1)(e)
    e = layers.Dense(32, name='Enc_32')(e)
    e = _leaky_relu(0.1)(e)
    e = layers.Dense(16, name='Enc_16')(e)
    e = _leaky_relu(0.1)(e)
    e = layers.Dense(8, name='Enc_8')(e)
    e = _leaky_relu(0.1)(e)

    bottleneck = layers.Dense(1, activation='linear', name='Bottleneck_1D')(e)

    d = layers.Dense(8, name='Dec_8')(bottleneck)
    d = _leaky_relu(0.1)(d)
    d = layers.Dense(16, name='Dec_16')(d)
    d = _leaky_relu(0.1)(d)
    d = layers.Dense(32, name='Dec_32')(d)
    d = _leaky_relu(0.1)(d)
    d = layers.Dense(48, name='Dec_48')(d)
    d = _leaky_relu(0.1)(d)
    output_layer = layers.Dense(input_dim, activation='sigmoid', name='Salida_Reconstruida')(d)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

    val_split = 0.1 if len(X) >= 10 else 0.0
    historial = autoencoder.fit(
        X, X, epochs=8, batch_size=256, shuffle=True,
        validation_split=val_split, verbose=1
    )
    acc_key = 'val_binary_accuracy' if 'val_binary_accuracy' in historial.history else 'binary_accuracy'
    accuracy_final = historial.history[acc_key][-1] * 100
    print(f"  [+] Autoencoder completado. Precisión reconstrucción: {accuracy_final:.1f}%")

    encoder_solo = Model(inputs=input_layer, outputs=bottleneck)
    score_ae = encoder_solo.predict(X, batch_size=512)

    col_ae = f'{col_target}_Autoencoder_1D'
    df_resultado[col_ae] = score_ae[:, 0]
    return df_resultado, [col_ae]


def main():
    parser = argparse.ArgumentParser(description="Pipeline de correlaciones y reducción dimensional.")
    parser.add_argument("-i", "--input", required=True,
                        help="Archivo de entrada en formato CSV, TSV o .sc (ej. metricas.csv)")
    parser.add_argument("-c", "--column", required=True,
                        help="Columna sobre la que aplicar las reducciones (ej. hotspots)")
    parser.add_argument("-t", "--tasks", nargs='+', choices=['heurísticas', 'heuristicas', 'pca', 'autoencoder'],
                        default=['heurísticas', 'pca'],
                        help="Métodos a aplicar (por defecto: heurísticas pca)")
    parser.add_argument("-m", "--method", choices=['spearman', 'pearson', 'kendall'], default='spearman',
                        help="Método de correlación (por defecto: spearman)")
    parser.add_argument("--metrics", nargs='+', default=None,
                        help="Métricas adicionales a incluir en el análisis (opcional)")
    parser.add_argument("-n", "--name", default=None,
                        help="Nombre base para los archivos de salida")

    args = parser.parse_args()

    if not args.name:
        input_stem = Path(args.input).stem
        args.name = f"correlaciones_{input_stem}_{args.method}"

    args.tasks = ['heurísticas' if t == 'heuristicas' else t for t in args.tasks]

    print(f"[*] Correlación ({args.method.upper()}) sobre '{args.column}' | Tareas: {args.tasks}")

    try:
        input_path_str = str(args.input)
        if input_path_str.endswith('.tsv'):
            df = pd.read_csv(args.input, sep='\t')
        elif input_path_str.endswith('.sc'):
            df = pd.read_csv(args.input, sep=r'\s+', engine='python')
            for col in ('SCORE:', 'SEQUENCE:'):
                if col in df.columns:
                    df = df.drop(columns=[col])
        elif input_path_str.endswith('.csv'):
            df = pd.read_csv(args.input)
        else:
            try:
                df = pd.read_csv(args.input)
            except Exception:
                df = pd.read_csv(args.input, sep=r'\s+', engine='python')
        print(f"[*] Diseños cargados: {len(df)}")
    except FileNotFoundError:
        print(f"[!] ERROR: Archivo no encontrado: {args.input}")
        sys.exit(1)

    if args.column not in df.columns:
        print(f"[!] ERROR: La columna '{args.column}' no existe en el archivo.")
        sys.exit(1)

    col_clean = f'{args.column}_tokens'

    def _tokenize(v):
        if pd.isna(v):
            return set()
        s = str(v).replace(' ', '')
        if s in ('', 'nan'):
            return set()
        return {t for t in s.split(',') if t}

    df[col_clean] = df[args.column].apply(_tokenize)

    elementos_unicos = set()
    for tokens in df[col_clean]:
        elementos_unicos.update(tokens)

    columnas_binarias = []
    for item in sorted(elementos_unicos):
        col_binaria = f'bin_{args.column}_{item}'
        df[col_binaria] = df[col_clean].apply(lambda toks: int(item in toks))
        columnas_binarias.append(col_binaria)

    if not columnas_binarias:
        print(f"[!] ERROR: La columna '{args.column}' no contiene valores válidos.")
        sys.exit(1)

    print(f"[+] Elementos únicos detectados: {len(columnas_binarias)}")

    columnas_nuevas_totales = []

    if 'heurísticas' in args.tasks:
        df, cols = calcular_descriptores_heuristas(df, args.column, columnas_binarias)
        columnas_nuevas_totales.extend(cols)

    if 'pca' in args.tasks:
        df, cols = aplicar_reduccion_pca(df, args.column, columnas_binarias)
        columnas_nuevas_totales.extend(cols)

    if 'autoencoder' in args.tasks:
        df, cols = aplicar_reduccion_autoencoder(df, args.column, columnas_binarias)
        columnas_nuevas_totales.extend(cols)

    if not columnas_nuevas_totales:
        print("[!] ERROR: Ninguna tarea generó nuevas columnas para correlación.")
        sys.exit(1)

    ratio_col = None
    if 'pae_plddt_ratio' in df.columns:
        ratio_col = 'pae_plddt_ratio'
    elif 'Macro_CUTRE' in df.columns:
        ratio_col = 'Macro_CUTRE'
    elif 'score_cutre' in df.columns:
        ratio_col = 'score_cutre'

    metricas_base = [
        col for col in [
            ratio_col, 'pae_interaction', 'plddt_binder', 'plddt_target', 'plddt_total', 'ddG',
            'Shape_complementarity', 'dSASA', 'Packstat', 'interface_hbonds',
            'interface_unsat_hbonds', 'SAP', 'binder_int_hyd', 'binder_surf_hyd',
            'binder_aligned_rmsd', 'binder_length',
            'self_complex_i_pTM', 'self_complex_i_pAE', 'self_complex_avg_ipSAE',
            'self_complex_pLDDT', 'self_binder_scRMSD', 'self_complex_scRMSD'
        ] if col is not None
    ]
    if args.metrics:
        for m in args.metrics:
            if m not in metricas_base:
                metricas_base.append(m)

    metricas_clave = metricas_base + columnas_nuevas_totales

    columnas_presentes = [col for col in metricas_clave if col in df.columns]
    df_num = df[columnas_presentes].copy()

    for col in df_num.columns:
        df_num[col] = pd.to_numeric(df_num[col], errors='coerce')

    n_antes = df_num.shape[1]
    df_num = df_num.dropna(axis=1, how='all')
    df_num = df_num.loc[:, df_num.nunique(dropna=True) > 1]
    descartadas = n_antes - df_num.shape[1]
    if descartadas:
        print(f"[*] {descartadas} columna(s) invariantes o vacías descartadas.")

    if df_num.shape[1] < 2:
        print("[!] ERROR: Quedan menos de 2 columnas válidas para correlación.")
        sys.exit(1)

    corr_matrix = df_num.corr(method=args.method)

    out_csv = f"{args.name}.csv"
    corr_matrix.to_csv(out_csv)
    print(f"[+] Matriz guardada en: {out_csv}")

    _load_plotting()
    plt.figure(figsize=(18, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .7}
    )

    plt.title(f"Correlación ({args.method.capitalize()}) - '{args.column}'", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_png = f"{args.name}.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"[+] Gráfico guardado en: {out_png}")


if __name__ == '__main__':
    main()
