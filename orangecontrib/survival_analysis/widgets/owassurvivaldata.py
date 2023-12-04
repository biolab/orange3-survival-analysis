from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.data import (
    Table,
    Domain,
    ContinuousVariable,
    TimeVariable,
    StringVariable,
    DiscreteVariable,
)

from orangecontrib.survival_analysis.widgets.data import (
    TIME_VAR,
    EVENT_VAR,
    TIME_TO_EVENT_VAR,
    get_survival_endpoints,
)


class OWAsSurvivalData(OWWidget):
    name = 'As Survival Data'
    description = 'Mark features Time and Event as target variables.'
    icon = 'icons/owassurvivaldata.svg'
    priority = 0
    want_main_area = False
    resizing_enabled = False
    keywords = ['time', 'event', 'censoring']

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        data = Output('Data', Table)

    settingsHandler = DomainContextHandler()
    time_var = ContextSetting(None)
    event_var = ContextSetting(None)
    auto_commit: bool = Setting(True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data: Optional[Table] = None

        time_var_model = DomainModel(valid_types=(ContinuousVariable,))
        event_var_model = DomainModel(valid_types=(DiscreteVariable,))

        box = gui.vBox(self.controlArea, 'Time', margin=0)
        gui.comboBox(
            box,
            self,
            'time_var',
            model=time_var_model,
            callback=self.commit.deferred,
            searchable=True,
        )

        box = gui.vBox(self.controlArea, 'Event', margin=0)
        gui.comboBox(
            box,
            self,
            'event_var',
            model=event_var_model,
            callback=self.commit.deferred,
            searchable=True,
        )

        self.commit_button = gui.auto_commit(
            self.controlArea, self, 'auto_commit', '&Commit', box=False
        )

    @Inputs.data
    def set_data(self, data: Table) -> None:
        self.closeContext()
        self._data = None
        domain: Optional[Domain] = None

        if data:
            # shallow copy data table and table attributes
            self._data = data.transform(data.domain)
            self._data.attributes = data.attributes.copy()
            # look for survival data in meta and class vars only.
            metas = [
                var
                for var in data.domain.metas
                if not isinstance(var, (TimeVariable, StringVariable))
            ]
            class_vars = [
                var
                for var in data.domain.class_vars
                if not isinstance(var, (TimeVariable, StringVariable))
            ]

            domain = Domain([], metas=metas, class_vars=class_vars)

        self.controls.time_var.model().set_domain(domain)
        self.controls.event_var.model().set_domain(domain)

        time_var_model = self.controls.time_var.model()
        event_var_model = self.controls.event_var.model()

        # If not found in the domain then default to the first var in model.
        _time_var, _event_var = get_survival_endpoints(domain)

        if len(time_var_model):
            self.time_var = time_var_model[0] if _time_var is None else _time_var

        if len(event_var_model):
            self.event_var = event_var_model[0] if _event_var is None else _event_var

        # Lastly, respect saved domain context
        if self.time_var is not None and self.event_var is not None:
            self.openContext(domain)

        self.commit.now()

    def as_survival_data(self, data: Table) -> Optional[Table]:
        if not self.time_var or not self.event_var or not data:
            return

        class_vars = [self.time_var, self.event_var]
        time_var = self.time_var
        event_var = self.event_var
        time_var.attributes[TIME_TO_EVENT_VAR] = TIME_VAR
        event_var.attributes[TIME_TO_EVENT_VAR] = EVENT_VAR

        metas = [meta for meta in data.domain.metas if meta not in class_vars]
        domain = Domain(data.domain.attributes, metas=metas, class_vars=class_vars)
        data = data.transform(domain)
        return data

    @gui.deferred
    def commit(self) -> None:
        self.Outputs.data.send(self.as_survival_data(self._data))


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWAsSurvivalData).run(
        Table('http://datasets.biolab.si/core/melanoma.tab')
    )
