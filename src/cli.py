import argparse

from .data_processor import load_and_process_campaigns, aggregate_by_channel, get_best_channel_by_roas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI de marketing analytics + RAG: responde perguntas simples sobre desempenho de canais."
    )

    parser.add_argument(
        "--pergunta",
        type=str,
        required=False,
        default="Qual canal teve melhor ROAS?",
        help='Pergunta em linguagem natural (ex: "Qual canal teve melhor ROAS?").',
    )

    args = parser.parse_args()
    question = args.pergunta.lower()

    df = load_and_process_campaigns()
    by_channel = aggregate_by_channel(df)

    if "roas" in question and "canal" in question:
        channel, roas = get_best_channel_by_roas(by_channel)
        print(f"O canal com melhor ROAS foi: {channel} (ROAS = {roas:.2f}).")
    else:
        print("Pergunta nao suportada ainda. Exemplos: 'Qual canal teve melhor ROAS?'")


if __name__ == "__main__":
    main()
