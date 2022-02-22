from functools import wraps

from Orange.widgets.utils.messages import UnboundMsg
from Orange.data import Table


TIME_COLUMN = 'time_var'
EVENT_COLUMN = 'event_var'
PROBLEM_TYPE = 'time_to_event'


def check_survival_data(f):
    """Check for survival data."""
    error_msg = 'No survival data detected. Use the "As Survival Data" widget or consult the documentation.'

    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Error.add_message('missing_survival_data', UnboundMsg(error_msg))
        widget.Error.missing_survival_data.clear()

        if data is not None and isinstance(data, Table):

            if not all(label in data.attributes for label in [TIME_COLUMN, EVENT_COLUMN]):
                widget.Error.missing_survival_data()
                data = None

        return f(widget, data, *args, **kwargs)

    return wrapper
