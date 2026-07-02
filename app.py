import subprocess
import tempfile
import os
import gradio as gr


def rank_candidates(file):
    output = tempfile.NamedTemporaryFile(
        suffix=".csv",
        delete=False
    ).name

    subprocess.run([
        "python",
        "rank.py",
        "--candidates",
        file.name,
        "--out",
        output,
        "--top",
        "100"
    ], check=True)

    return output


demo = gr.Interface(
    fn=rank_candidates,
    inputs=gr.File(label="Upload candidates.jsonl"),
    outputs=gr.File(label="Download Ranked CSV"),
    title="AI Candidate Discovery & Ranking",
    description="""
Upload a candidate dataset (.jsonl).
The ranking engine analyzes each candidate and generates a ranked CSV.
"""
)

if __name__ == "__main__":
    demo.launch()