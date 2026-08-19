import base64

import numpy as np
import diskcache
import umap
import plotly.graph_objects as go
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, html, dcc, DiskcacheManager, ctx
import dash

#from data.dataset import images_labels_from_hierarchie
from planktolab.models.backbone import create_model, get_available_models, save_model, load_model
from planktolab.utils import get_features_probas_labels, generate_default_path_name
from planktolab.pipeline import run_train, run_train_kfolds, run_resize_images, run_detect_suspect, run_inference


def create_app():

    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

    app = Dash(__name__, background_callback_manager=background_callback_manager)
    app.layout = dmc.MantineProvider(layout)
    return app

# Settings for clean labels and train are similar, but differ in the specific settings. 
# This function creates the specific settings for each case.
def create_model_setting(clean_labels):
    if clean_labels:
        specific_setting = dmc.NumberInput(
                label="K-Fold (for suspect detection)",
                min=2,
                w=250,
                id="kfold-input",
                value=5,
            )
    else:
        specific_setting = dmc.Group([
            dmc.TextInput(label="Validation folder", placeholder="Validation folder", id="val-folder"),
            dmc.TextInput(label="Test folder", placeholder="Test folder", id="test-folder"),

        ], id="val-test-folders-input")
    
    submit_button = "kfold-button" if clean_labels else "train-button"

    return html.Div([dmc.Group(children=[
            dmc.TextInput(label="Output folder (will be created)", placeholder="Output folder", value=generate_default_path_name(), id="output-folder"),
            dmc.TextInput(label="Train folder", placeholder="Input folder", id="train-folder"),
            specific_setting
        ]),

        dmc.Group(children=[
            dmc.Select(
                data=[{"value": model, "label": model} for model in get_available_models()],
                placeholder="Select a model",
                id="model-select",
                label="Model"
            ),
            dmc.NumberInput(
                label="Batch size",
                min=1,
                max=2048,
                id="batch-size-input",
                value=64
            ),
            dmc.NumberInput(
                label="Max epochs",
                min=1,
                id="max-epoch",
                value=20
            ),
        ]),

        dmc.Button("Train", id=submit_button, color="blue", mt="md"),
    ], id="model-setting")
    
# Resize images setting
prepare_data_setting = html.Div(
    children=[
        dmc.TextInput(label="Input folder", placeholder="Input folder", id="input-folder-prepare"),
        dmc.TextInput(label="Output folder", placeholder="Output folder", value="", id="output-folder-prepare"),
        dmc.NumberInput(
            label="Image size",
            min=32,
            max=1024,
            w=250,
            id="image-size-input",
            value=128
        ),
        dmc.Button("Resize Images", id="resize-button", color="blue", mt="md"),


    ], id="prepare-data-setting"
)


# Settings for detecting suspects
detect_suspect_setting = dmc.Fieldset(
    children=[
        dmc.Group(children=[

            dmc.TextInput(label="kfold folder", placeholder="Folder that contains kfold results", id="input-folder-detect"),

            dmc.Select(
                    data=[
                        {"value": "self_confidence", "label": "Self Confidence"},
                        {"value": "knn", "label": "K Nearest Neighbors"},
                    ],
                    placeholder="Select a method",
                    id="method-select",
                    label="Method"
                ),
        ]),
        dmc.NumberInput(
            label="Threshold (for suspect detection)",
            min=0.0,
            max=1.0,
            step=0.01,
            w=250,
            id="threshold-input",
            value=0.5
        ),
        dmc.Button("Detect Suspects", id="detect-button", color="blue", mt="md"),

    ], 
    variant="filled",
	radius="sm",
	disabled=False,
    id="detect-suspect-setting"
)



plot_images = dmc.Fieldset(
    children=[
        dmc.Group(children=[
            dmc.TextInput(label="Input folder", placeholder="Input folder", id="input-folder-plot"),
            dmc.Select(
                data=[],
                placeholder="Select a fold",
                id="fold-select-plot",
                disabled=True,
                label="Fold"
            )
        ]),
        dmc.Button("Plot Images", id="plot-button", color="blue", mt="md"),
        dcc.Graph(
            id="umap-graph",
            #figure=fig,
            style={"flex": "2"}
        ),
        html.Div([
            html.Img(
                id="image-display",
                style={
                    "marginLeft": "20px",
                    "maxWidth": "400px",
                    "height": "auto",
                    "flex": "1"
                }
            ),
            html.Img(id="mask-display", style={"marginTop": "20px", "maxWidth": "400px", "height": "auto"})
        ],style={
            "display": "flex",
            "flexDirection": "column",
            "marginLeft": "20px"
        })

    ], 
    variant="filled",
	radius="sm",
	disabled=False,
    id="plot-images-setting"
)


detect_visualise_suspect_setting = dmc.Tabs(
    # props as configured above:
    children=[
        dmc.TabsList([
            dmc.TabsTab("Detect Suspects", value="suspects", disabled=False),
            dmc.TabsTab("Visualise", value="visualise", disabled=False),
        ]),
        dmc.TabsPanel(detect_suspect_setting, value="suspects"),
        dmc.TabsPanel(plot_images, value="visualise"),
    ],
    variant="default",
    radius="md",

)


inference_setting = html.Div([
    dmc.Group([
        dmc.TextInput(label="Model folder", placeholder="Folder that contains the trained model", id="input-folder-inference"),
        dmc.TextInput(label="Input folder", placeholder="Folder that contains the images to run inference on", id="input-folder-inference-images"),
        dmc.Button("Run Inference", id="inference-button", color="blue", mt="md"),
    ], id="inference-content"),
])

layout = dmc.AppShell(
    [
        dmc.AppShellHeader(
            dmc.Group(
                [
                    dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
                    #dmc.Image(src=logo, h=40),
                    dmc.Title("PlanktonLab", c="blue"),
                ],
                h="100%",
                px="md",
            )
        ),
        dmc.AppShellNavbar(
            id="navbar",
            children=[
                "Navbar",
                dmc.NavLink(label="Prepare Data", href="/prepare", id="link-prepare"),
                dmc.NavLink(label="Train", href="/train", id="link-train"),
                dmc.NavLink(label="Clean Labels", href="/clean-labels", id="link-cleanlabels"),
                dmc.NavLink(label="Inference", href="/inference", id="link-inference"),
            ],
            p="md",
        ),
        dmc.AppShellMain([dmc.Container([
            dcc.Store(id="log-file", data=""),
            dcc.Store(id="model-running", data=False),
            dcc.Location(id="url"),
            dmc.Title("Train", id="title"),
            dmc.Text("This is the train page. Here you can train your model.", id="description"),
            dmc.Fieldset(
                variant="filled",
                radius="sm",
                disabled=False,
                id="settings",
                # other props...
            ),


            dmc.Button("Cancel", color="red", mt="md", id="train-cancel"),

            html.Pre(id="log-output", style={
                "height": "400px",
                "overflowY": "scroll",
                "backgroundColor": "black",
                "color": "lime",
                "padding": "10px"
            }),
            html.Div(id="result"),

            #detect_visualise_suspect_setting,


            dcc.Interval(id="log-interval", interval=2000),
            html.Div(id="trash")

        ], fluid=True)],id="main-content"),
    ],
    header={"height": 60},
    padding="md",
    navbar={
        "width": 300,
        "breakpoint": "sm",
        "collapsed": {"mobile": True},
    },
    id="appshell",

)




@callback(
    Output("appshell", "navbar"),
    Input("burger", "opened"),
    State("appshell", "navbar"),
)
def navbar_is_open(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar

@callback(
    Output("settings", "children"),
    Output("result", "children", allow_duplicate=True),

    Output("title", "children"),
    Output("description", "children"),
    Input("url", "pathname"),
    prevent_initial_call=True,

)
def display_page(pathname):
    if pathname == "/prepare":
        return prepare_data_setting, [], "Prepare Data", "This is the prepare data page. Here you can prepare your data for training."
    elif pathname == "/clean-labels":
        return create_model_setting(True), detect_visualise_suspect_setting, "Clean Labels", "This is the clean labels page. Here you can clean your labels."
    elif pathname == "/inference":
        return inference_setting, [], "Inference", "This is the inference page. Here you can run inference on your model."
    else:
        return create_model_setting(False), [], "Train", "This is the train page. Here you can train your model."


@callback(
    Output("log-file", "data"),
    Input("output-folder", "value"),
    prevent_initial_call=True,
    optional=True
)
def update_log_file(output_folder):
    return f"{output_folder}/log.txt"


# ----------- Inference callback -----------
@callback(
    Output("result", "children"),
    Input("inference-button", "n_clicks"),
    State("input-folder-inference", "value"),
    State("input-folder-inference-images", "value"),
    State("output-folder-inference", "value"),
    prevent_initial_call=True,
    optional=True
)
def run_inference_callback(n_clicks, model_folder, input_folder, output_folder):
    if n_clicks is None:
        return dash.no_update

    print("starting inference")
    labels, classes = run_inference(model_folder, input_folder, output_path=output_folder)
    u,counts = np.unique(labels, return_counts=True)
    graph = dcc.Graph(
        figure=go.Figure(
            data=[go.Bar(x=classes[u], y=counts)],
            layout=go.Layout(title="Inference Results", xaxis_title="Class", yaxis_title="Count")
        )
    )
    print("finished inference")
    return graph, dmc.Alert(f"Inference completed! Check the output folder {output_folder} for results.", color="green")


# ------------- Train callback -------------
@callback(
    Output("trash", "children", allow_duplicate=True),
    Input("train-button", "n_clicks"),
    State("model-select", "value"),
    State("train-folder", "value"),
    State("val-folder", "value"),
    State("test-folder", "value"),
    State("output-folder", "value"),
    State("max-epoch", "value"),
    State("batch-size-input", "value"),


    prevent_initial_call=True,
    running=[
        (Output("settings", "disabled"), True, False),
        (Output("model-running", "data"), True, False),
        ],
    cancel=Input("train-cancel", "n_clicks"),
    background=True,
    optional=True
)
def train_model(n_clicks, model_name, train_folder, val_folder, test_folder, output_folder, max_epoch, batch_size):
    if n_clicks is None:
        return dash.no_update

    run_train(train_folder, model_name, output_folder, max_epoch, batch_size, val_path=val_folder, test_path=test_folder)

    return dmc.Alert(f"Training started! Check the output folder {output_folder} for results.", color="green")


# ------------ Resize Images callback -----------
@callback(
    Output("trash", "children"),
    Input("resize-button", "n_clicks"),
    State("input-folder-prepare", "value"),
    State("output-folder-prepare", "value"),
    State("image-size-input", "value"),

    prevent_initial_call=True,
    optional=True
)
def resize_images(n_clicks, input_folder, output_folder, image_size):
    if n_clicks is None:
        return dash.no_update

    run_resize_images(input_folder, output_folder, image_size)

    return dmc.Alert(f"Images resized! Check the output folder {output_folder} for results.", color="green")



@callback(
    Output("mask-display", "src"),
    Input("umap-graph", "clickData"),
    prevent_initial_call=True,
    optional=True

)
def display_mask(clickData):
    if clickData is None:
        return ""
    
        
    img_path = clickData["points"][0]["customdata"]

    encoded_image = base64.b64encode(open(img_path, 'rb').read())

    return f"data:image/png;base64,{encoded_image}"


@callback(
    Output("umap-graph", "figure"),
    Input("fold-select-plot", "value"),
    Input("input-folder-plot", "value"),
    prevent_initial_call=True,
    optional=True
)
def compute_umap(fold_value, input_folder):

    i = fold_value

    features = np.load(f"{input_folder}/fold_{i}/features.npy")
    labels = np.load(f"{input_folder}/fold_{i}/labels.npy")
    probas = np.load(f"{input_folder}/fold_{i}/probas.npy")
    fname = np.load(f"{input_folder}/fold_{i}/fname.npy", allow_pickle=True)

    class_names = np.load(f"{input_folder}/fold_{i}/classes.npy", allow_pickle=True)

    #X_scaled = StandardScaler().fit_transform(features)

    feat_umap = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_components=2
    ).fit_transform(features)

    new_fig = go.Figure()

    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if not np.any(mask):
            continue

        new_fig.add_trace(
            go.Scattergl(
                x=feat_umap[mask, 0],
                y=feat_umap[mask, 1],
                customdata=fname[mask],
                mode="markers",
                name=class_name,
                marker=dict(size=4),
            )
        )

    new_fig.update_layout(width=1200, height=900)

    return new_fig


@callback(
    Output("log-output", "children"),
    Input("log-interval", "n_intervals"),
    State("log-file", "data"),
    State("model-running", "data")
)
def update_logs(n, log_file, model_running):
    if not model_running:
        return "Model is not running."

    try:
        with open(log_file, "r") as f:
            return f.read()[-5000:]  # limite taille
    except:
        return "No logs yet..."

def main():
    app = create_app()
    app.run(debug=True)

if __name__ == "__main__":
    main()