"""Prepara a planilha de erros (E3) para anotação rápida.

Lê a planilha gerada por `jev-eval error-sheet` e o `raw_predictions.jsonl` da mesma execução e grava
uma cópia ordenada por par de confusão (rótulo certo -> previsto), com colunas de apoio calculadas
por regras determinísticas. A coluna `category` continua em branco: quem decide é o grupo.

Colunas de apoio (inseridas antes de `category`; o `error-summary` ignora colunas extras):
- par, linhas_no_par: o par de confusão e quantas linhas da planilha o compartilham;
- erro_tambem_em_C1: se o mesmo sistema errou o mesmo item em inglês (sim / nao / sem_resposta);
- traducao_possivel: "sim" só quando o sistema acertou em C1 e errou aqui (única situação em que a
  tradução pode ser a causa);
- outros_erram_igual: quantos outros sistemas escolheram o mesmo rótulo errado neste item;
- outro_sistema_anotado: o outro sistema da planilha que também errou este item (e o que previu);
- dica: lembretes derivados das colunas acima (não é a categoria);
- bloco: divisão em 4 blocos por pares inteiros, equilibrada por número de itens;
- concordancia: "sim" nos itens que todos devem rotular, para medir a concordância (kappa).

Uso:
  python scripts/prepare_error_annotation.py \
      --sheet outputs/final/banking77/error_analysis/error_annotation_C3.csv \
      --raw outputs/final/raw_predictions.jsonl
Grava <planilha>_apoio.csv ao lado da original. Com --substituir, sobrescreve a original (só se
nenhuma categoria estiver preenchida), para o `error-summary` ler a versão anotada.

Divisão entre integrantes: --por-bloco PASTA grava bloco_1.csv ... bloco_4.csv (linhas do bloco e,
ao final, os itens de concordância dos outros blocos). Depois de preenchidos:
  python scripts/prepare_error_annotation.py --juntar PASTA
grava a planilha original com a categoria do dono de cada bloco, concordancia.csv na PASTA e o kappa
de Fleiss dos itens de concordância.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

N_BLOCKS = 4
N_AGREEMENT = 20
SEED = 42
SUPPORT = ["par", "linhas_no_par", "erro_tambem_em_C1", "traducao_possivel", "outros_erram_igual",
           "outro_sistema_anotado", "dica", "bloco", "concordancia"]


def read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader.fieldnames or []), list(reader)


def load_answers(path: Path) -> dict[tuple[str, str, str, str], dict]:
    """(dataset, condition, example_id, system) -> resposta válida."""
    answers = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("error") is None:
                answers[(r["dataset"], r["condition"], r["example_id"], r["system"])] = r
    return answers


def assign_blocks(pairs: dict[str, list[dict]]) -> dict[str, int]:
    """Pares inteiros em N_BLOCKS blocos, do maior para o menor, sempre no bloco com menos itens."""
    load = [set() for _ in range(N_BLOCKS)]
    block_of = {}
    for pair, rows in sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        items = {r["example_id"] for r in rows}
        target = min(range(N_BLOCKS), key=lambda b: (len(load[b] | items), b))
        block_of[pair] = target + 1
        load[target] |= items
    return block_of


def hint(row: dict[str, str]) -> str:
    notes = []
    if row["erro_tambem_em_C1"] == "sim":
        notes.append("também erra em EN: tradução improvável")
    if row["traducao_possivel"] == "sim":
        notes.append("acerta em EN: comparar text com text_en")
    if row["outros_erram_igual"] == "3":
        notes.append("todos erram igual: conferir se o rótulo de referência é defensável")
    if int(row["linhas_no_par"]) >= 3:
        notes.append("par frequente: decidir uma vez para o par e procurar exceções")
    return "; ".join(notes)


def prepare(fields: list[str], rows: list[dict[str, str]], answers: dict) -> tuple[list[str], list[dict]]:
    if any(r.get("category", "").strip() for r in rows):
        raise SystemExit("A planilha já tem categorias preenchidas; não vou reorganizá-la.")
    systems_in_sheet = {r["system"] for r in rows}
    all_systems = sorted({k[3] for k in answers})
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r = dict(r)
        dataset, condition, example_id, system = r["dataset"], r["condition"], r["example_id"], r["system"]
        r["par"] = f"{r['true_label']} -> {r['predicted_label']}"
        if condition == "C1":
            r["erro_tambem_em_C1"], r["traducao_possivel"] = "(planilha de C1)", "nao"
        else:
            en = answers.get((dataset, "C1", example_id, system))
            r["erro_tambem_em_C1"] = "sem_resposta" if en is None else ("nao" if en["correct"] else "sim")
            r["traducao_possivel"] = "sim" if en is not None and en["correct"] else "nao"
        others = [answers.get((dataset, condition, example_id, s)) for s in all_systems if s != system]
        r["outros_erram_igual"] = str(sum(1 for o in others
                                          if o and o["predicted_label_name"] == r["predicted_label"]))
        mates = [f"{s} ({answers[(dataset, condition, example_id, s)]['predicted_label_name']})"
                 for s in sorted(systems_in_sheet - {system})
                 if (dataset, condition, example_id, s) in answers
                 and not answers[(dataset, condition, example_id, s)]["correct"]]
        r["outro_sistema_anotado"] = ", ".join(mates)
        by_pair[r["par"]].append(r)

    block_of = assign_blocks(by_pair)
    item_ids = sorted({r["example_id"] for r in rows})
    agreement = set(random.Random(SEED).sample(item_ids, min(N_AGREEMENT, len(item_ids))))
    out = []
    for pair, pair_rows in sorted(by_pair.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        for r in sorted(pair_rows, key=lambda x: (x["example_id"], x["system"])):
            r["linhas_no_par"] = str(len(pair_rows))
            r["bloco"] = str(block_of[pair])
            r["concordancia"] = "sim" if r["example_id"] in agreement else ""
            r["dica"] = hint(r)
            out.append(r)
    cut = fields.index("category")
    return fields[:cut] + SUPPORT + fields[cut:], out


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_blocks(directory: Path, header: list[str], rows: list[dict]) -> None:
    """bloco_<b>.csv: as linhas do bloco e, ao final, as de concordância dos outros blocos."""
    for b in range(1, N_BLOCKS + 1):
        own = [r for r in rows if r["bloco"] == str(b)]
        extra = [dict(r, concordancia="sim (outro bloco)") for r in rows
                 if r["bloco"] != str(b) and r["concordancia"] == "sim"]
        write_csv(directory / f"bloco_{b}.csv", header, own + extra)
        print(f"gravado: {directory / f'bloco_{b}.csv'} ({len(own)} linhas do bloco + {len(extra)} de concordância)")


def fleiss_kappa(labels_per_row: list[list[str]], categories: list[str]) -> float | None:
    rows = [labels for labels in labels_per_row if len(labels) >= 2]
    if not rows or len({len(labels) for labels in rows}) != 1:
        return None
    n = len(rows[0])
    counts = [[labels.count(c) for c in categories] for labels in rows]
    p_bar = sum((sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts) / len(counts)
    totals = [sum(row[j] for row in counts) / (len(counts) * n) for j in range(len(categories))]
    p_e = sum(p * p for p in totals)
    return None if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def join_blocks(directory: Path, sheet: Path, out: Path) -> None:
    """Junta os blocos preenchidos: a categoria final de cada linha é a do dono do bloco."""
    _, original = read_sheet(sheet)
    if out == sheet and any(r.get("category", "").strip() for r in original):
        raise SystemExit(f"{sheet} já tem categorias; use --saida para gravar em outro arquivo.")
    final: dict[tuple[str, str], dict] = {}
    agreement: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    header: list[str] = []
    for b in range(1, N_BLOCKS + 1):
        header, rows = read_sheet(directory / f"bloco_{b}.csv")
        for r in rows:
            key = (r["example_id"], r["system"])
            if r["bloco"] == str(b):
                final[key] = r
            if r["concordancia"].startswith("sim"):
                agreement[key][f"bloco_{b}"] = r["category"].strip()
    missing = [(r["example_id"], r["system"]) for r in original if (r["example_id"], r["system"]) not in final]
    if missing:
        raise SystemExit(f"{len(missing)} linhas da planilha original não estão nos blocos (ex.: {missing[:3]}).")
    blank = [k for k, r in final.items() if not r["category"].strip()]
    if blank:
        print(f"atenção: {len(blank)} linhas ainda sem categoria (ex.: {blank[:3]})")
    rows = [final[(r["example_id"], r["system"])] for r in original]
    write_csv(out, header, rows)
    print(f"gravado: {out} ({len(rows)} linhas)")

    names = [f"bloco_{b}" for b in range(1, N_BLOCKS + 1)]
    table = [{"example_id": k[0], "system": k[1], **{n: v.get(n, "") for n in names}}
             for k, v in sorted(agreement.items())]
    write_csv(directory / "concordancia.csv", ["example_id", "system", *names], table)
    filled = [[v[n] for n in names] for v in agreement.values() if all(v.get(n) for n in names)]
    categories = sorted({c for labels in filled for c in labels})
    kappa = fleiss_kappa(filled, categories)
    print(f"concordância: {len(filled)} linhas rotuladas pelos {N_BLOCKS} blocos; kappa de Fleiss = "
          + ("n/d" if kappa is None else f"{kappa:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", type=Path,
                        default=Path("outputs/final/banking77/error_analysis/error_annotation_C3.csv"))
    parser.add_argument("--raw", type=Path, default=Path("outputs/final/raw_predictions.jsonl"))
    parser.add_argument("--out", type=Path, help="padrão: <planilha>_apoio.csv")
    parser.add_argument("--substituir", action="store_true", help="sobrescreve a planilha original")
    parser.add_argument("--por-bloco", type=Path, metavar="PASTA",
                        help="também grava bloco_1.csv ... bloco_4.csv nesta pasta")
    parser.add_argument("--juntar", type=Path, metavar="PASTA",
                        help="junta os blocos preenchidos desta pasta na planilha (ou em --saida)")
    parser.add_argument("--saida", type=Path, help="com --juntar: arquivo de saída (padrão: a planilha original)")
    args = parser.parse_args()

    if args.juntar:
        join_blocks(args.juntar, args.sheet, args.saida or args.sheet)
        return

    fields, rows = read_sheet(args.sheet)
    header, out = prepare(fields, rows, load_answers(args.raw))
    path = args.sheet if args.substituir else (args.out or args.sheet.with_name(args.sheet.stem + "_apoio.csv"))
    write_csv(path, header, out)
    if args.por_bloco:
        split_blocks(args.por_bloco, header, out)

    items = {r["example_id"] for r in out}
    blocks = Counter()
    for b in range(1, N_BLOCKS + 1):
        blocks[b] = len({r["example_id"] for r in out if r["bloco"] == str(b)})
    print(f"gravado: {path}")
    print(f"{len(out)} linhas, {len(items)} itens, {len({r['par'] for r in out})} pares")
    print(f"tradução possível: {sum(r['traducao_possivel'] == 'sim' for r in out)} linhas; "
          f"todos erram igual: {sum(r['outros_erram_igual'] == '3' for r in out)} linhas")
    print("itens por bloco: " + ", ".join(f"{b}: {n}" for b, n in sorted(blocks.items())))
    print(f"itens de concordância (todos rotulam): {sum(1 for i in items if any(r['concordancia'] and r['example_id'] == i for r in out))}")


if __name__ == "__main__":
    main()
