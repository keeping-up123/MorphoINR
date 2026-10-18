import argparse
from omegaconf import OmegaConf


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        choices=['embedder', 'inr_film'],
        required=True
    )
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument(
        '--case',
        type=str,
        default=None,
        help='Optional case identifier overriding the value in the YAML config.',
    )
    args = parser.parse_args(argv)

    from utils import pprint_config

    configs = OmegaConf.load(args.conf)
    if args.case is not None:
        configs.case = args.case
    print(args.conf)
    pprint_config(configs)

    if args.mode == "embedder":
        from systems import train_embedder
        train_embedder(configs)
    elif args.mode == "inr_film":
        from systems import train_inr_film
        train_inr_film(configs)

    pprint_config(configs)
    return configs


if __name__ == "__main__":
    main()
