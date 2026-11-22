"""
app.py
======
Interactive Gradio Web Application for Enterprise IT Helpdesk
Ticket Intent Classification & Automated Routing Simulator.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import gradio as gr
except ImportError:
    print("Gradio is not installed. Install with: pip install gradio")
    sys.exit(1)

LABELS = [
    "PASSWORD_RESET",
    "ACCESS_ISSUE",
    "HARDWARE_FAILURE",
    "SOFTWARE_INSTALL",
    "NETWORK_LATENCY",
    "GENERAL_QUERY"
]

ROUTING_INFO = {
    "PASSWORD_RESET": {
        "icon": "🔑",
        "team": "Identity & Access Management (IAM)",
        "sla": "15 Minutes",
        "action": "Trigger automated self-service password reset workflow."
    },
    "ACCESS_ISSUE": {
        "icon": "🔒",
        "team": "Systems & Directory Administration (Tier 2)",
        "sla": "2 Hours",
        "action": "Verify user groups in Active Directory & VPN security profile."
    },
    "HARDWARE_FAILURE": {
        "icon": "🖥️",
        "team": "On-Site Hardware Services & Asset Desk",
        "sla": "4 Hours",
        "action": "Schedule physical inspection or issue replacement equipment."
    },
    "SOFTWARE_INSTALL": {
        "icon": "📦",
        "team": "Desktop Engineering & Software Packaging",
        "sla": "1 Business Day",
        "action": "Validate enterprise software licensing and push installation via MDM."
    },
    "NETWORK_LATENCY": {
        "icon": "🌐",
        "team": "Network Operations Center (NOC)",
        "sla": "1 Hour",
        "action": "Inspect switch port status, Wi-Fi AP saturation, and packet routes."
    },
    "GENERAL_QUERY": {
        "icon": "❓",
        "team": "IT Service Desk Tier 1",
        "sla": "8 Hours",
        "action": "Provide relevant Knowledge Base article and FAQ reference."
    }
}


def classify_ticket(ticket_text: str):
    if not ticket_text or not ticket_text.strip():
        return {}, "Please provide a ticket description."

    text_lower = ticket_text.lower()

    prediction_source = "BERT + LoRA"
    # Model inference if checkpoints exist, otherwise rule-based confidence simulator
    adapter_path = "checkpoints/bert-lora-helpdesk"
    use_model = os.path.exists(adapter_path)

    if use_model:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=len(LABELS)
            )
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()

            inputs = tokenizer(ticket_text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze().tolist()

            scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        except Exception as error:
            print(f"[WARNING] Model inference failed: {error}")
            scores = simulate_scores(text_lower)
            prediction_source = "DEMO MODE (rule-based)"
    else:
        scores = simulate_scores(text_lower)
        prediction_source = "DEMO MODE (rule-based)"

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label] * 100.0

    route = ROUTING_INFO.get(best_label, {})
    routing_html = f"""
    <div style="padding: 16px; border-radius: 8px; border: 1px solid #e0e0e0; background-color: #f9fafb; color: #1f2937;">

        <p style="color: #1f2937; margin-top: 0;">
            <strong style="color: #1f2937 !important;">Prediction Source:</strong> {prediction_source}
        </p>

        <h3 style="margin-top: 0; color: #1e3a8a;">
            {route.get('icon', '')} Intent: {best_label} ({best_score:.1f}% Score)
        </h3>

        <p style="color: #1f2937;">
            <strong style="color: #1f2937 !important;">🏢 Automated Routing Target:</strong> {route.get('team', 'Unassigned')}
        </p>

        <p style="color: #1f2937;">
            <strong style="color: #1f2937 !important;">⏱️ Target Resolution SLA:</strong> {route.get('sla', 'Standard')}
        </p>

        <p style="color: #1f2937;">
            <strong style="color: #1f2937 !important;">⚡ Action Plan:</strong> {route.get('action', 'Manual review')}
        </p>

    </div>
    """

    return scores, routing_html


def simulate_scores(text: str):
    scores = {l: 0.05 for l in LABELS}
    if any(k in text for k in ["password", "credential", "locked", "expire", "login"]):
        scores["PASSWORD_RESET"] = 0.85
    elif any(k in text for k in ["vpn", "permission", "access", "folder", "forbidden", "403"]):
        scores["ACCESS_ISSUE"] = 0.85
    elif any(k in text for k in ["monitor", "laptop", "keyboard", "mouse", "screen", "fan", "battery", "crashed"]):
        scores["HARDWARE_FAILURE"] = 0.85
    elif any(k in text for k in ["install", "license", "software", "docker", "vs code", "chrome", "app"]):
        scores["SOFTWARE_INSTALL"] = 0.85
    elif any(k in text for k in ["wifi", "wi-fi", "internet", "latency", "ping", "packet", "slow", "down"]):
        scores["NETWORK_LATENCY"] = 0.85
    else:
        scores["GENERAL_QUERY"] = 0.85

    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def build_app():
    custom_css = """
    .gradio-container { max-width: 900px !important; margin: auto; }
    """

    with gr.Blocks(title="IT Helpdesk AI - LoRA BERT", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🤖 Enterprise IT Helpdesk Intent Classifier
            ### Fine-Tuned on BERT-Base (110M Parameters) using LoRA (~1.2 MB Adapter)
            Type an IT issue below or click one of the pre-loaded ticket examples.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                ticket_input = gr.Textbox(
                    label="Enter IT Support Request",
                    placeholder="e.g. My Windows password expired and I cannot log in...",
                    lines=3
                )
                submit_btn = gr.Button("Classify & Route Ticket", variant="primary")

                gr.Examples(
                    examples=[
                        ["I forgot my corporate Windows password and need an urgent unlock."],
                        ["Getting a 403 Forbidden permission error when attempting to open the Z: network drive."],
                        ["External monitor went completely dark and laptop fan is overheating."],
                        ["Requesting installation and developer license for Docker Desktop."],
                        ["The conference room Wi-Fi connection keeps dropping with high latency."],
                        ["What is the standard procedure for ordering ergonomic desk equipment?"]
                    ],
                    inputs=ticket_input,
                    label="Quick Ticket Examples"
                )

            with gr.Column(scale=2):
                label_output = gr.Label(num_top_classes=3, label="Classification Scores")
                routing_output = gr.HTML(label="Automated Dispatch Details")

        submit_btn.click(
            fn=classify_ticket,
            inputs=ticket_input,
            outputs=[label_output, routing_output]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
