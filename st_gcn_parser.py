import argparse
import json


class Parser(argparse.ArgumentParser):
    """Custom Parser to read CLI arguments, JSON config files, and add parameters to namespace.

    Methods:
        parse_args(args, namespace)
            Hierarchicaly combines CLI and config file parameters in the namespace.

        json_to_args(dict)
            Converts dictionary to a formated argparse string.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Sets up the parser for the ST-GCN script."""

        super().__init__(*args, **kwargs)
        

    def parse_args(self, args=None, namespace=None):
        """Parses CLI arguments first, then defaults to the config file.
        
        CLI arguments take precedence, parameters inside the config file provide default
        values for arguments that the user did explicitly set.

        Args:
            args : ``list[str]``
                String of CLI arguments.

            namespace : ``Namespace``
                Namespace object to add the parsed arguments to.

        Returns:
            Namespace object with parsed arguments. Convert to a ``dict`` by ``vars(obj)``.
        """
        
        # get the CLI arguments
        # will yield default --config file if no arguments provided
        cli_args = super().parse_args(args, namespace)

        # parse the config file, which will set parameters not provided
        # by user as CLI arguments
        with open(cli_args.config, 'r') as f:
            data = json.load(f)
        
        # parse arguments in the file and store them in a blank namespace
        config_args = dict([items for sub_dict in data.items() for items in sub_dict[1].items()])

        # set arguments in the target namespace if they haven’t been set yet
        for k, v in config_args.items():
            if getattr(cli_args, k, None) is None:
                setattr(cli_args, k, v)
        
        return cli_args


    def json_to_args(self, data: dict) -> str:
        """Converts dictionary to a formated argparse string.
        
        Builds a single string by prepending each key of the dictionary with a ``--`` 
        and concatenates values depending on the type:
            flag parameters
                just the key if true
            single-value parameters
                key and value
            list parameters
                key and unpacked list
            2D list parameters
                same key passed with each unpacked sublist

        Args:
            data : ``dict``
                Python dictionary with CLI arguments as keys.

        Returns:
            String with correctly formatted keys and values to be passed to the parser.
        """
        
        arg = ''
        for group in data:
            for k, v in data[group].items():
                if type(v) is list and type(v[0]) is not list:
                    arg += ' --'+k+' '+' '.join([str(e) for e in v])
                elif type(v) is list and type(v[0]) is list:
                    for e in v:
                        arg += ' --'+k+' '+' '.join([str(ee) for ee in e])
                elif type(v) is bool and v is True:
                    arg += ' --'+k
                elif type(v) is bool and v is False:
                    continue
                else:
                    arg += ' --'+k+' '+str(v)
        return arg
