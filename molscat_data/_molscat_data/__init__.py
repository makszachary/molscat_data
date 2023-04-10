import warnings

warnings.filterwarnings(
    action='ignore',
    message=r'.*significant figures requested from number with only.*',
    category=UserWarning,
    module=r'.*sigfig'
)