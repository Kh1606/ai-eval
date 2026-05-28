"""Render the EvalKit 'how it works' flowchart.

Run from this directory:  python howitworks.py
Deps:  pip install diagrams cairosvg   (and Graphviz on PATH)
"""
import os
from diagrams import Diagram, Cluster, Edge
from diagrams.generic.storage import Storage
from diagrams.custom import Custom

ICONS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "_diagram_icons"))


def icon(name: str) -> str:
    return os.path.join(ICONS, f"{name}.png")


graph_attr = {"fontsize": "18", "bgcolor": "white", "pad": "0.4", "splines": "spline"}
node_attr = {"fontsize": "13"}
edge_attr = {"fontsize": "11"}

with Diagram(
    "EvalKit — unified model evaluation CLI",
    filename="howitworks",
    direction="LR",
    show=False,
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
):
    with Cluster("Inputs (per run)"):
        model = Storage("Model\ncheckpoint")
        data = Storage("Test\ndataset")

    cli = Custom("evalkit\ninteractive CLI", icon("python"))

    with Cluster("Evaluators  (repeated random 50% subsampling)"):
        classify = Custom("classify\nimage classification", icon("pytorch"))
        detect = Custom("detect\nobject detection", icon("ultralytics"))
        seg = Custom("seg\nsemantic segmentation", icon("opencv"))
        speech = Custom("speech\nASR (Wav2Vec2)", icon("pytorch"))
        text = Custom("text\nclassification", icon("pytorch"))
        evals = [classify, detect, seg, speech, text]

    reports = Custom("Reports\nCMs · ROC · PR\nhistograms · t-SNE", icon("plotly"))
    csv = Custom("CSV summary\nmetrics + stability", icon("files"))

    model >> Edge(color="#0ea5e9") >> cli
    data >> Edge(color="#10b981") >> cli
    cli >> Edge(color="#8b5cf6") >> evals
    evals >> Edge(color="#f59e0b", label="metrics") >> reports
    evals >> Edge(color="#f59e0b") >> csv
