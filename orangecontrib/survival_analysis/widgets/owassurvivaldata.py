from typing import Optional

from Orange.widgets import gui
from Orange.widgets.settings import Setting, ContextSetting, DomainContextHandler
from Orange.widgets.widget import Input, Output, OWWidget
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable, StringVariable, DiscreteVariable

from orangecontrib.survival_analysis.widgets.data import TIME_COLUMN, EVENT_COLUMN, PROBLEM_TYPE


class OWAsSurvivalData(OWWidget):
    name = 'As Survival Data'
    description = ''
    icon = 'icons/owassurvivaldata.svg'
    priority = 0
    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input('Data', Table)

    class Outputs:
        data = Output('Data', Table)

    settingsHandler = DomainContextHandler()
    time_var = ContextSetting(None, schema_only=True)
    event_var = ContextSetting(None, schema_only=True)
    auto_commit: bool = Setting(True, schema_only=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data: Optional[Table] = None

        time_var_model = DomainModel(valid_types=(ContinuousVariable,))
        event_var_model = DomainModel(valid_types=(DiscreteVariable,))

        box = gui.vBox(self.controlArea, 'Time', margin=0)
        gui.comboBox(box, self, 'time_var', model=time_var_model, callback=self.commit.deferred, searchable=True)

        box = gui.vBox(self.controlArea, 'Event', margin=0)
        gui.comboBox(box, self, 'event_var', model=event_var_model, callback=self.commit.deferred, searchable=True)

        self.commit_button = gui.auto_commit(self.controlArea, self, 'auto_commit', '&Commit', box=False)

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
            vars_ = [
                var
                for var in data.domain.metas + data.domain.class_vars
                if not isinstance(var, (TimeVariable, StringVariable))
            ]
            domain = Domain(vars_)

        self.controls.time_var.model().set_domain(domain)
        self.controls.event_var.model().set_domain(domain)

        self.time_var = None
        self.event_var = None
        self.openContext(domain)
        self.commit.now()

    def as_survival_data(self, data: Table) -> Optional[Table]:
        if not self.time_var or not self.event_var or not data:
            return

        metas = [meta for meta in data.domain.metas if meta not in (self.time_var, self.event_var)]
        domain = Domain(data.domain.attributes, metas=metas, class_vars=[self.time_var, self.event_var])
        data = data.transform(domain)
        data.attributes[TIME_COLUMN] = self.time_var
        data.attributes[EVENT_COLUMN] = self.event_var
        data.attributes['problem_type'] = PROBLEM_TYPE
        return data

    @gui.deferred
    def commit(self) -> None:
        self.Outputs.data.send(self.as_survival_data(self._data))


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWAsSurvivalData).run(Table('http://datasets.biolab.si/core/melanoma.tab'))
