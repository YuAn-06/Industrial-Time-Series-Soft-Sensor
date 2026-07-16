"""Template capability card for a model folder."""

from models.base import ModelSpec


MODEL_SPEC = ModelSpec(
    # Rename the copied folder and update these identity fields together.
    name="TemplateModel",
    module="models._template",

    # Declare only tasks that the architecture genuinely supports.
    supported_tasks=("soft_sensor", "short_term_forecasting"),

    # Describe the input data structure and loss protocol used by the model.
    dataset_type="standard",
    loss_type="default",

    # Example for a two-stage model: ("pretrain",). Keep empty otherwise.
    pretrain_stages=(),

    # Documentation fields are optional.
    paper_title="",
    paper_url="",
    source_url="",
)
