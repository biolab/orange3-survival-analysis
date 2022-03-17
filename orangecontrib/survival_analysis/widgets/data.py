from functools import wraps
from typing import Tuple, Optional

from Orange.widgets.utils.messages import UnboundMsg
from Orange.data import Table, Domain, Variable


TIME_VAR = 'time'
EVENT_VAR = 'event'
TIME_TO_EVENT_VAR = '_time_to_event_var'


def contains_survival_endpoints(domain: Domain):
    class_vars = domain.class_vars
    return (
        len(class_vars) == 2
        and all(TIME_TO_EVENT_VAR in t.attributes for t in class_vars)
        and all(t.attributes[TIME_TO_EVENT_VAR] in [TIME_VAR, EVENT_VAR] for t in class_vars)
    )


def get_survival_endpoints(domain: Domain) -> Tuple[Optional[Variable], Optional[Variable]]:
    time_var = None
    event_var = None
    if contains_survival_endpoints(domain):
        class_vars = domain.class_vars
        for var in class_vars:
            surv_var_type = var.attributes[TIME_TO_EVENT_VAR]
            if surv_var_type == TIME_VAR:
                time_var = var
            elif surv_var_type == EVENT_VAR:
                event_var = var
    return time_var, event_var


def check_survival_data(f):
    """Check for survival data."""
    error_msg = 'No survival data detected. Use the "As Survival Data" widget or consult the documentation.'
    warning_msg = 'Rows with missing values detected. They will be omitted.'

    @wraps(f)
    def wrapper(widget, data: Table, *args, **kwargs):
        widget.Error.add_message('missing_survival_data', UnboundMsg(error_msg))
        widget.Error.missing_survival_data.clear()

        widget.Warning.add_message('missing_values_detected', UnboundMsg(warning_msg))
        widget.Warning.missing_values_detected.clear()

        if data is not None and isinstance(data, Table):
            if not contains_survival_endpoints(data.domain):
                widget.Error.missing_survival_data()
                data = None
            elif data.has_missing():
                widget.Warning.missing_values_detected()

        return f(widget, data, *args, **kwargs)

    return wrapper
